# coding=utf-8
# Copyleft 2020 project COL.

from collections import defaultdict
import math
import pickle
import os
import sys

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

import common
from indexing import TorchGPUIndexer, FaissGPUIndexer

VERY_LARGE = 9595959595


class Vokenizer:
    def __init__(self, model, tokenizer, keys_dir, img_sets=('coco_minival',),
                 max_img_num=VERY_LARGE, gpus=(0,), backend='faiss', upper_bound=128,
                 sent_level=False):
        """

        :param model: Hugginface language model
        :param tokenizer: Hugginface Tokenizer
        :param keys_dir: the directory which saves the keys.
        :param img_sets: the img_sets to be loaded, see common.IMAGE_SETS for all options.
        :param max_img_num: load up to #max_img_num images into the dictionary
        :param gpus: The GPUs used in calculating the BERT outputs and indexing.
                     Note: Currently only one GPU is supported!!!
        """
        self.model = model.cuda(gpus[0]) if model is not None else model
        self.tokenizer = tokenizer
        self.img_sets = img_sets
        self.gpus = gpus        # The GPUs used in the indexer
        self.gpu = self.gpus[0]
        self.backend = backend
        self.upper_bound = upper_bound
        self.sent_level = sent_level    # Otherwise use word level

        max_img_num = VERY_LARGE if max_img_num == -1 else max_img_num
        # These two are important, which indicates the mapping from
        # vokens to their actual images.
        self.img_paths = []
        self.img_ids = []
        for img_set in self.img_sets:
            assert img_set in common.IMAGE_SETS, "%s not in image sets %s" % (
                img_set, common.IMAGE_SETS)

            # Load image paths corresponding to the keys.
            # img_paths_fname = os.path.join(common.LOCAL_DIR, 'images', img_set + "_paths.txt")
            # img_ids_fname = os.path.join(common.LOCAL_DIR, 'images', img_set + "_ids.txt")
            img_paths_fname = os.path.join(keys_dir, f"{img_set}.path")
            img_ids_fname = os.path.join(keys_dir, f"{img_set}.ids")
            if not os.path.exists(img_paths_fname):
                # If the actual images are not saved on the server, we would use the img_ids.
                img_paths_fname = img_ids_fname
            with open(img_paths_fname) as f:
                all_img_paths = list(map(lambda x: x.strip(), f.readlines()))
            with open(img_ids_fname) as g:
                all_img_ids = list(map(lambda x: x.strip(), g.readlines()))
            assert len(all_img_paths) == len(all_img_ids)
            for img_path, img_id in zip(all_img_paths, all_img_ids):
                if len(self.img_paths) < max_img_num:
                    self.img_paths.append(img_path)
                    self.img_ids.append(f"{img_set}/{img_id}")
                else:
                    break
        assert len(self.img_paths) == len(self.img_ids)

        # Lazy loading and indexing
        self.keys = None
        self.keys_dir = keys_dir
        self.indexed = False
        self.indexer = None

    @property
    def img_num(self):
        return len(self.img_paths)

    def dump_img_ids(self, fname):
        """
        Dump the mapping from the voken_id to img_ids, to fname.
        Saved in the format of array.
        """
        with open(fname, 'w') as f:
            for img_id in self.img_ids:
                f.write(img_id + "\n")

    def __len__(self):
        return self.img_num

    def indexing(self):
        self.model.eval()

        # Load pre-extracted image keys.
        self.keys = []
        remain_img_num = self.img_num
        for img_set in self.img_sets:
            assert img_set in common.IMAGE_SETS, "%s not in image sets %s" % (
                img_set, common.IMAGE_SETS)
            keys_fname = os.path.join(self.keys_dir, img_set + '.hdf5')
            if not os.path.exists(keys_fname):
                assert False, "keys of image set %s is not extracted, please save it at %s" % (
                    img_set, keys_fname
                )

            # Load Keys
            h5_file = h5py.File(keys_fname, 'r')
            dset = h5_file["keys"]
            load_img_num = min(remain_img_num, len(dset))
            load_keys = dset[:load_img_num]
            self.keys.append(load_keys)
            remain_img_num -= load_img_num
            h5_file.close()
            if load_img_num == 0:
                break

        # Lazy indexing
        self.keys = np.concatenate(self.keys, 0)
        if self.backend == 'torch':
            self.indexer = TorchGPUIndexer(self.keys, gpus=self.gpus, fp16=True)
        elif self.backend == 'faiss':
            self.indexer = FaissGPUIndexer(self.keys, gpus=self.gpus, fp16=True)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported")

        self.indexed = True

    def vokenize_sents(self, sents, topk=None):

        input_ids = []
        for sent in sents:
            input_ids.append(self.tokenizer.encode(
                sent,
                add_special_tokens=False,
                # return_tensors='pt'     # Return PyTorch (pt) tensors
            ))
        return self.vokenize_ids(input_ids, attention_mask=None, topk=topk)

    def vokenize_ids(self, input_ids, attention_mask=None, topk=None):
        """
        :param input_ids:  A list of token_ids i.e.,
                [[token_1_1, token_1_2, ...], [token_2_1, token_2_2, ...], ...]
        :param attention_mask: I did not use it for now.
        :param topk: Retrieve the topk vokens for each token.
        :return: top_scores, top_idxs, input_tokens, top_paths
            Note: 1. The results would consider the additional special tokens while the input_tokens do **not**.
                  2. If topk=None, it will be a 2-d results with:
                         [ [s11_top1, s12_top1, ...],
                           [s21_top1, s22_top1, ...],
                           ..... ]
                     If topk!=None (e.g., 1, 5, 10), it will be a 3-d results with:
                         [ [ [s11_top1, s11_top2, ...],
                             [s12_top1, s12_top2, ...],
                             ...... ],
                           [ [s21_top1, s21_top2, ...],
                             [s22_top1, s22_top2, ...],
                             ...... ],
                           ..... ],
                    where s11_top1 means s1(the 1st sentence)1(the 1st token of the 1st sentence)_top1(the top-1 index)
        """
        if not self.indexed:        # Index the keys at the first retrieval call.
            self.indexing()

        # The original tokens
        input_tokens = [
            ([self.tokenizer.cls_token] + [self.tokenizer._convert_id_to_token(idx) for idx in input_id] + [self.tokenizer.sep_token])
            for input_id in input_ids]

        # Deal with over-length tokens (because the BERT-style encoder has length limit due to the positional embedding)
        # Here is a process to avoid very short sequence when cutting the long sentence:
        # Suppose the sentence length is 18 and UPPER_BOUND is 8,
        # we draw it as                         <----------------->, where "<" is bos, and ">" is the last token
        # instead of cut it as                  <------->------->->, which has very short sequence <-> in the end.
        # we cut it with almost equal length:   <----->----->----->
        input_ids = input_ids.copy()
        sent2segs = defaultdict(list)
        for i in range(len(input_ids)):
            if len(input_ids[i]) > self.upper_bound:
                num_segments = math.ceil(len(input_ids[i]) / self.upper_bound)
                tokens_per_seg = int(len(input_ids[i]) / num_segments)
                remaining = input_ids[i][tokens_per_seg:]
                input_ids[i] = input_ids[i][:tokens_per_seg]
                while len(remaining) > 0:
                    # print(len(remaining))
                    sent2segs[i].append(len(input_ids))
                    input_ids.append(remaining[:tokens_per_seg])
                    remaining = remaining[tokens_per_seg:]

        # Convert to torch tensors.
        if not type(input_ids) is torch.Tensor:
            input_ids = [
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(list(input_id)))
                for input_id in input_ids
            ]
            input_ids = pad_sequence(input_ids,
                                     batch_first=True,
                                     padding_value=self.tokenizer.pad_token_id)
            attention_mask = (input_ids != self.tokenizer.pad_token_id)         # word_tokens --> 1, pad_token --> 0
            if attention_mask.all():
                attention_mask = None

        # Get lengths
        if attention_mask is not None:
            lengths = list(attention_mask.sum(1).numpy())
        else:
            lengths = [len(input_ids[0])] * len(input_ids)

        if attention_mask is not None and type(input_ids) is not torch.Tensor:
            attention_mask = torch.tensor(attention_mask)

        # Lang model inference
        input_ids = input_ids.cuda(self.gpu)
        if attention_mask is not None:
            attention_mask = attention_mask.cuda(self.gpu)

        def apply_model(input_ids, attention_mask, lengths):
            with torch.no_grad():
                lang_output = self.model(input_ids, attention_mask)     # b, l, f
                if type(lang_output) is list:
                    lang_output = lang_output[0]

            # Gather language output
            if self.sent_level:
                # lang_output of shape [batch_size, dim]
                gathered_output = lang_output
            else:
                # lang_output of shape [batch_size, max_len, dim]
                # --> gathered_output [ \sum_i len(i), dim]
                gathered_output = torch.cat([output[:length] for output, length in zip(lang_output, lengths)])

            # Visn retrieval
            if topk is None:
                # It will call the function `max()` and return a 2-d tensor
                top_score, top_idx = self.indexer.batch_top1(gathered_output)
            else:
                # It will call the function `topk(k)` and return a 3-d tensor
                top_score, top_idx = self.indexer.batch_topk(gathered_output, topk=topk)

            return top_score, top_idx

        top_score, top_idx = memory_safe_apply(apply_model, input_ids, attention_mask, lengths)

        # Split
        top_score, top_idx = top_score.detach().cpu(), top_idx.detach().cpu()
        if not self.sent_level:
            # If word level, split it
            top_scores = list(top_score.split(lengths))       # [ float_tensor(len1), float_tensor(len2), ...]
            top_idxs = list(top_idx.split(lengths))           # [ int_tensor(len1), int_tensor(len2), ...]
        else:
            # If sent level, repeat the voken.
            #   Use clone() here
            top_scores = [ts.expand(length, *ts.shape).clone() for ts, length in zip(top_score, lengths)]
            top_idxs = [tid.expand(length, *tid.shape).clone() for tid, length in zip(top_idx, lengths)]

        if top_idxs[0].dim() == 1:
            # Return the top1 paths
            top_paths = [[self.img_paths[idx.item()] for idx in top_idx]
                         for top_idx in top_idxs]
        else:
            # Return the topk paths related to the sentences
            top_paths = [[[self.img_paths[k_idx.item()] for k_idx in topk_idx]
                          for topk_idx in top_idx]
                         for top_idx in top_idxs]

        if self.sent_level:
            for i, tid in enumerate(top_idxs):
                # Keep the first positive and others negative, to mark the header of the sentence.
                # [3] --> [3, 3, 3, 3] --> [-4, -4, -4, -4] --> [3, -4, -4, -4]
                # "-x-1" is used to handle zero, [0] --> [1, 1, 1, 1] --> [-1, -1, -1, -1] --> [0, -1, -1, -1]
                # print('Before conversion', tid)
                tid[:] = tid * (-1) - 1
                tid[1] = tid[1] * (-1) - 1  # The tid[0] is corresponding to <cls>
                # print('After conversion', top_idxs[i])

        # Put back the segments of over-length sentences
        if len(sent2segs) > 0:
            for sent_id, segment_ids in sent2segs.items():
                for segment_id in segment_ids:
                    # Append the results with the segments:
                    #    ---------Now----------------   + ----Appended Segment-----
                    #    [<cls1> I have a <sep1>][:-1]  + [<cls2> cat . <sep2>][1:]
                    #  = [<cls1> I have a cat . <sep2>]
                    top_scores[sent_id] = torch.cat([top_scores[sent_id][:-1], top_scores[segment_id][1:]])
                    top_idxs[sent_id] = torch.cat([top_idxs[sent_id][:-1], top_idxs[segment_id][1:]])
                    top_paths[sent_id] = top_paths[sent_id][:-1] + top_paths[segment_id][1:]
            num_sents = len(input_tokens)
            top_scores = top_scores[:num_sents]
            top_idxs = top_idxs[:num_sents]
            top_paths = top_paths[:num_sents]

        return top_scores, top_idxs, input_tokens, top_paths


def memory_safe_apply(func, *args):
    """
    If batch-wise applying exceeds the GPU memory, it would process each sample separately and sequentially
    :param func: function with some constraints, see code for details.
    :param args: args of this function
    :return:
    """
    try:
        return func(*args)
    except RuntimeError as e:
        print(e)
        batch_size = len(args[0])
        outputs = []
        for i in range(batch_size):
            one_batch_args = tuple(a[i: i+1] for a in args)
            output = func(*one_batch_args)
            # **output of the func should be of the format**:
            # (o1, o2, ...) where each o_i is a tensor of shape [1, ...]
            assert type(output) is tuple or type(output) is list
            outputs.append(output)
    # outputs = ( (o1_1, o1_2, ...), (o2_1, o2_2, ...), ...)
    # zip(*outputs) = ( (o1_1, o2_1, ...), (o1_2, o2_2, ...), ...)
    outputs = tuple(torch.cat(output) for output in zip(*outputs))
    return outputs


default_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_model_and_tokenizer(load, cpu=False):
    if os.path.exists(load + '/BEST.pth.model'):
        sys.path.append(load + '/src')
        for dirc in os.listdir(load + '/src'):
            sys.path.append(load + '/src/' + dirc)
        # import model  # The pickle has some issues... thus must load the library
        if cpu:
            device = torch.device('cpu')
            joint_model = torch.load(load + '/BEST.pth.model',
                                     map_location=device)
        else:
            joint_model = torch.load(load + '/BEST.pth.model')
        joint_model.eval()  # DO NOT FORGET THIS!!!
    else:
        print("No snapshots there, exit.")
        exit()

    if os.path.exists(load + '/tokenizer.pkl'):
        with open(load + '/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = default_tokenizer

    return joint_model.lang_model, tokenizer
