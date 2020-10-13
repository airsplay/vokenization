import argparse
from collections import defaultdict
import json
import os

import tqdm

from vokenization import Vokenizer, load_model_and_tokenizer
import common

imgset2fname = {
    'coco_train': 'mscoco_train.json',
    'coco_nominival': 'mscoco_nominival.json',
    'coco_minival': 'mscoco_minival.json',
    'vg_nococo': 'vg_nococo.json',
    'cc_train': 'training.tsv',
    'cc_valid': 'validation.tsv',
}


def load_cc_data(img_set):
    fname = os.path.join(common.CC_ROOT, imgset2fname[img_set])
    sentXimgname = []
    with open(fname) as f:
        for line in f:
            sent, gt_img_name = line.split('\t')
            gt_img_name = gt_img_name.strip()
            sentXimgname.append((sent, gt_img_name))
    print("Load the %d (img, sent) pairs for image set %s from %s" % (
        len(sentXimgname), img_set, fname))
    return sentXimgname


def load_lxrt_data(img_set):
    fname = os.path.join(common.LXRT_ROOT, imgset2fname[img_set])
    sentXimgname = []
    with open(fname) as f:
        data = json.load(f)
        for datum in data:
            gt_img_name = datum['img_id'] + '.jpg'
            sents = datum['sentf']['mscoco']
            for sent in sents:
                sentXimgname.append((sent, gt_img_name))
    print("Load the %d (img, sent) pairs for image set %s from %s" % (
        len(sentXimgname), img_set, fname))
    return sentXimgname


# load = '/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/coco_hinge05_dim64_resxt101_bertl4'
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/coco_hinge05_dim64_resxt101_robertal4',
                    help='The directory saved the model (containing'
                         'BEST.pth.model).')
parser.add_argument('--image-sets', type=str, default='coco_minival',
                    help='The splits of images to be extracted')
args = parser.parse_args()

keys_path = os.path.join(args.load, 'keys')

print("Evaluate for model %s on image sets %s" % (args.load, args.image_sets))
model, tokenizer = load_model_and_tokenizer(args.load)
img_sets = args.image_sets.split(',')

sent_level = 'sent' in args.load

for img_set in img_sets:
    vokenizer = Vokenizer(model, tokenizer, keys_path, [img_set],
                          sent_level=sent_level)
    if 'cc' in img_set:
        sentXimgname = load_cc_data(img_set)
    else:
        sentXimgname = load_lxrt_data(img_set)

    topks = [1, 5, 10]
    print("\nEvaluate image set", img_set, "for topk retrieval:", topks)
    total = 0
    arg_topk = None if max(topks) == 1 else max(topks)
    results = defaultdict(lambda: 0)
    batch_size = 32
    for start_id in tqdm.tqdm(range(0, len(sentXimgname), batch_size)):
        batch_sentXimg = sentXimgname[start_id: start_id + batch_size]
        sents, gt_img_names = zip(*batch_sentXimg)
        sents = list(sents)

        scores, ids, tokens, paths_list = vokenizer.vokenize_sents(sents, topk=arg_topk)
        if sent_level:
            paths_list = [x[:3] for x in paths_list]     # Only eval the first vokens.
        if arg_topk is None:
            paths_list = [[[img_id] for img_id in sent] for sent in paths_list]
        for paths, gt_img_name in zip(paths_list, gt_img_names):                # for each sent in batch
            for topk_paths in paths[1:-1]:      # for each token in sent
                for k, kth_path in enumerate(topk_paths):     # for each img_path in topk image paths of a token
                    img_name = os.path.split(kth_path)[-1]
                    if img_name == gt_img_name:
                        results[k + 1] += 1
        total += sum(map(lambda x: len(x) - 2, paths_list))

    accumulate = 0
    for i in range(1, max(topks)+1):
        accumulate += results[i]
        if i in topks:
            print("R%d: %0.2f%%, (Random: %0.4f%%)" % (
                i,
                accumulate / total * 100.,
                i / vokenizer.img_num * 100.
            ))

    del vokenizer




