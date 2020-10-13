import copy
import os
import random

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import tqdm


class CoLDataset(Dataset):
    IGNORE_ID = -100
    sent_strategy = 'first'

    def __init__(self, file_path, tokenizer_name, tokenizer, block_size=512,
                 split_sent=False, voken_dir=None, suffix=None, verbose=False,
                 voken_ablation=None):

        # Open token's hdf5
        token_path = file_path + '.' + tokenizer_name + '.hdf5'
        assert os.path.isfile(token_path)
        if verbose:
            print("-------- Load Data -------")
            print("Load tokens from", token_path)
        self.token_hdf5 = h5py.File(token_path, 'r')
        self.tokenizer = tokenizer
        self.tokens = self.token_hdf5['tokens']
        self.verbose = verbose
        self.voken_ablation = voken_ablation
        self._iter_cnt = 0

        # Open voken's hdf5 and load voken ids
        if voken_dir is not None:
            assert suffix is not None, 'Please provide suffix of the voken, e.g., vg_nococo.5000.'
            self.sent_level = 'sent' in voken_dir
            dset_fname = os.path.split(file_path)[-1]
            voken_path = os.path.join(voken_dir, f"{dset_fname}.{suffix}.hdf5")
            voken_ids_path = os.path.join(voken_dir, f"{dset_fname}.{suffix}.ids")
            if verbose:
                print("Load vokens from", voken_path)
            self.voken_hdf5 = h5py.File(voken_path, 'r')
            self.vokens = self.voken_hdf5['vokens']
            assert len(self.vokens) == len(self.tokens)
            self._voken_ids = list(
                map(lambda x: x.strip(),
                    open(voken_ids_path).readlines())
            )
            if verbose:
                print("\t with voken size", self.voken_size)
                print("\t top 5 voken ids are:", self._voken_ids[:5])
        else:
            self.vokens = None

        # Split for every block_size tokens
        # The last block without full length will be dropped.
        num_tokens = len(self.tokens)
        self.starts = list(range(0, num_tokens, block_size))
        self.batches = list(zip(self.starts[:-1], self.starts[1:]))

        manual_filtered =False
        if "en.train.raw" in file_path and tokenizer_name == "bert-base-uncased":
            self.batches = manual_filter(self.batches)
            if verbose:
                print("Data: Mannually filter the range for counties.")
            manual_filtered = True

        # batch_info
        if verbose:
            print("Split sent with block size", block_size)
            print(f"Total batches: {len(self.batches)}")
            print(f"Total tokens: {len(self.tokens)}")
            if voken_dir is not None:
                print(f"Total vokens: {len(self.vokens)}")
            if voken_ablation is not None:
                print("The model will process voken ablation strategy:", voken_ablation)
            print()

        block_check(self.batches, block_size, fixed_size=True, manual_filtered=manual_filtered)
        if self.voken_ablation == 'token':
            self._voken_ids = list(range(30522))

    @property
    def voken_size(self):
        return len(self._voken_ids)

    @property
    def voken_ids(self):
        return copy.copy(self._voken_ids)

    def assert_equal_vokens(self, dataset):
        assert self.voken_size == dataset.voken_size
        for vid, vid1 in zip(self.voken_ids, dataset.voken_ids):
            assert vid == vid1

    def __len__(self):
        return len(self.batches) - 1

    def __getitem__(self, item):
        token_start, token_end = self.batches[item]
        if self._iter_cnt < 5 and self.verbose:
            print(f"Data Loader: data iteration {self._iter_cnt}, with range {token_start} to {token_end}.")
            self._iter_cnt += 1
        tokens = list(self.tokens[token_start: token_end])
        token_tensor = torch.tensor(
            self.tokenizer.build_inputs_with_special_tokens(tokens),
            dtype=torch.long)
        if self.vokens is not None:
            vokens = list(self.vokens[token_start: token_end])

            vokens = self.maybe_do_sent_level(vokens)
            vokens = self.maybe_do_ablation_study(vokens, tokens)

            voken_tensor = torch.tensor(
                [self.IGNORE_ID] + vokens + [self.IGNORE_ID],
                dtype=torch.long
            )

            return token_tensor, voken_tensor
        else:
            return token_tensor

    def maybe_do_sent_level(self, vokens):
        if not self.sent_level:
            return vokens
        else:
            if self.sent_strategy == 'all':
                vokens = [
                    (-voken-1 if voken < 0 else voken)
                    for voken in vokens
                ]
            elif self.sent_strategy == 'first':
                vokens = [
                    (self.IGNORE_ID if voken < 0 else voken)
                    for voken in vokens
                ]
            return vokens

    def maybe_do_ablation_study(self, vokens, tokens):
        if self.voken_ablation is None:
            return vokens
        else:
            if self._iter_cnt < 5 and self.verbose:
                print("Before voken ablation: ", vokens)
            if self.voken_ablation == 'random':
                vokens = [random.randint(0, self.voken_size - 1)
                          for _ in range(len(vokens))]
            elif self.voken_ablation == 'shuffle':
                random.shuffle(vokens)
            elif self.voken_ablation == 'reverse':
                vokens = vokens[::-1]
            elif self.voken_ablation == 'token':
                vokens = tokens
            if self._iter_cnt < 5 and self.verbose:
                print("After voken ablation: ", vokens)
            return vokens

    def get_item_info(self, item):
        token_start = self.batches[item]
        token_end = self.batches[item + 1]
        return token_start, token_end

    def __del__(self):
        self.token_hdf5.close()
        if self.vokens is not None:
            self.voken_hdf5.close()


FORBIDDEN_RANGE = (
    119314944,      # Start of iter 3700
    187053048       # End of iter 5800
)


def intersect(x, y):
    x1, x2 = x
    y1, y2 = y
    if x2 <= y1 or x2 >= y2:
        # Case 1: [   x    )[   y    )
        # Case 2: [   y    )[   x    )
        return False
    return True


def manual_filter(batches):
    batches = list(filter(
        lambda x: not intersect(x, FORBIDDEN_RANGE),
        batches
    ))
    return batches


def block_check(batches, block_size, fixed_size=False, manual_filtered=False):
    """
    Check whether the batches satisfy following requirements.
        1. Monotonic
        2. Mutually exclusive
        3. Range < block_size
    """
    last_end = 0
    for start_token, end_token in batches:
        assert last_end <= start_token
        if fixed_size:
            assert (end_token - start_token) == block_size, 'len([%d, %d)) != %d' % (start_token, end_token, block_size)
        else:
            assert (end_token - start_token) <= block_size, 'len([%d, %d)) > %d' % (start_token, end_token, block_size)
        if manual_filtered:
            assert not intersect((start_token, end_token), FORBIDDEN_RANGE)
        last_end = end_token


def get_voken_feats(dataset: CoLDataset, feat_dir: str):
    """
    Load pre-extracted visual features regarding img_ids of vokens.
    """
    set2id2feat = {}
    voken_feats = []
    for voken_id in dataset.voken_ids:
        voken_img_set, voken_img_id = voken_id.split('/')
        if voken_img_set not in set2id2feat:
            img_ids = list(map(
                lambda x: x.rstrip(),
                open(os.path.join(feat_dir, f"{voken_img_set}.ids"))
            ))
            img_feats = h5py.File(
                os.path.join(feat_dir, f"{voken_img_set}.hdf5"), 'r'
            )['keys'][:]
            id2feat = {}
            assert len(img_ids) == len(img_feats)
            for img_id, img_feat in zip(img_ids, img_feats):
                id2feat[img_id] = img_feat
            set2id2feat[voken_img_set] = id2feat
        voken_feats.append(set2id2feat[voken_img_set][voken_img_id])
    return voken_feats



