import argparse
from collections import defaultdict
import json
import os
import sys

import numpy as np
import tqdm

from vokenization import Vokenizer, load_model_and_tokenizer
import common

imgset2fname = {
    'coco_train': 'mscoco_train.json',
    'coco_nominival': 'mscoco_nominival.json',
    'coco_minival': 'mscoco_minival.json',
    'vg_nococo': 'vgnococo.json',
    'cc_train': 'training.tsv',
    'cc_valid': 'validation.tsv',
}

tokenizer_name = 'bert-base-uncased'


def load_lang_data(corpus_name, topk=10000):
    """
    Load {topk} sentences from the corpus named by {corpus_name}.
    """
    fpath = corpus_name + '.' + tokenizer_name
    tokens = []
    with open(fpath) as f:
        for i, line in enumerate(f):
            tokens.append(list(map(int, line.split(' '))))
            if (i + 1) == topk:
                break
    print("Read %d sentences from the corpus %s located at %s." % (
        len(tokens), corpus_name, fpath
    ))
    return tokens


def load_cc_data(img_set):
    fname = os.path.join(common.CC_ROOT, imgset2fname[img_set])
    sents = []
    with open(fname) as f:
        for line in f:
            sent, _ = line.split('\t')
            sents.append(sent)
    print("Load the %d sentences for image set %s from %s" % (
        len(sents), img_set, fname))
    return sents


def load_lxrt_data(img_set):
    fname = os.path.join(common.LXRT_ROOT, imgset2fname[img_set])
    sents = []
    with open(fname) as f:
        data = json.load(f)
        for datum in data:
            sents.extend(datum['sentf']['mscoco'])
    print("Load the %d sentences for image set %s from %s" % (
        len(sents), img_set, fname))
    return sents


def analyze(token2info):
    """
    :param token2info: token2info: token --> (img_id --> cnt)
    :return:
    """
    names = ['Num Images', 'Max Cnt', 'Avg Cnt', 'Std Cnt']
    results = np.zeros(4)
    num_tokens = 0
    for token in token2info:
        img2cnt = token2info[token]
        cnts = np.array(list(img2cnt.values()))
        num_imgs = len(cnts)
        max_cnt = cnts.max()
        avg_cnt = cnts.mean()
        std_cnt = cnts.std()
        results += (num_imgs, max_cnt, avg_cnt, std_cnt)
        num_tokens += 1
    print("With %d tokens, " % num_tokens)
    results /= num_tokens
    for name, result in zip(names, results):
        print("Average of %s is %0.2f" % (name, result))

    corpus_info = defaultdict(lambda: 0)
    for info in token2info.values():
        for img, cnt in info.items():
            corpus_info[img] += cnt
    print("Cover %d images" % len(corpus_info))

# load = '/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/coco_hinge05_dim64_resxt101_bertl4'
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/coco_hinge05_dim64_resxt101_robertal4',
                    help='The directory saved the model (containing'
                         'BEST.pth.model).')
parser.add_argument('--image-sets', type=str, default='coco_minival',
                    help='The splits of images to be extracted')
parser.add_argument('--corpus', type=str, default='wiki103',
                    help='Evaluated corpus')
parser.add_argument('--maxsents', type=int, default=10000,
                    help='The maximum sentences to be evaluated in the corpus')
args = parser.parse_args()

keys_path = os.path.join(args.load, 'keys')

print("Evaluate for model %s on image sets %s" % (args.load, args.image_sets))
model, tokenizer = load_model_and_tokenizer(args.load)
img_sets = args.image_sets.split(',')
vokenizer = Vokenizer(model, tokenizer, keys_path, img_sets)

corpus_list = args.corpus.split(',')
for corpus in corpus_list:
    corpus = corpus.strip()
    print("\nProcessing corpus %s for diversity test:" % corpus)
    # token2info: token --> (img_id --> cnt)
    token2info = defaultdict(lambda: defaultdict(lambda: 0))

    if corpus in imgset2fname:
        if 'cc' in corpus:
            sents = load_cc_data(corpus)
        else:
            sents = load_lxrt_data(corpus)
        batch_size = 32
        for start_id in tqdm.tqdm(range(0, len(sents), batch_size)):
            batch_sents = sents[start_id: start_id + batch_size]
            scores, ids, tokens, paths = vokenizer.vokenize_sents(batch_sents, topk=None)
            for i in range(len(paths)):
                for token, path in zip(tokens[i][1:-1], paths[i][1:-1]):
                    token2info[token][path] += 1
    else:
        tokens_list = load_lang_data(corpus, args.maxsents)
        batch_size = 16
        for start_id in tqdm.tqdm(range(0, len(tokens_list), batch_size)):
            batch_tokens = tokens_list[start_id: start_id + batch_size]
            scores, ids, tokens, paths = vokenizer.vokenize_ids(batch_tokens, topk=None)
            for i in range(len(paths)):
                for token, path in zip(tokens[i][1:-1], paths[i][1:-1]):
                    token2info[token][path] += 1

    analyze(token2info)




