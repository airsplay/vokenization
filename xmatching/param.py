# coding=utf-8
# Copyleft 2020 project COL.
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        # print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        # print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        # print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        # print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--sources", default='mscoco', help="mscoco, cc, vg, vqa, gqa, visual7w")
    parser.add_argument("--train-imgs", default='mscoco_train,mscoco_nominival,vg_nococo')
    parser.add_argument("--valid-imgs", default='mscoco_minival')
    parser.add_argument("--train-langs", default='mscoco',
                        help='Some of mscoco, cc, vg, vqa, gqa, visual7w.'
                             'split by comma')
    parser.add_argument("--valid-langs", default='mscoco',
                        help='Some of mscoco, cc, vg, vqa, gqa, visual7w.'
                             'split by comma')
    parser.add_argument("--test", default=None)
    parser.add_argument("--test-only", action='store_true')

    # Datasets Configuration
    parser.add_argument("--fast", action='store_true')
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--max-len", default=20, type=int)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')
    parser.add_argument("--fp16", action='store_true')

    # Model Hyper-parameters
    parser.add_argument('--visn', type=str, default='resnext101_32x8d', help='The vision backbone model.')
    parser.add_argument('--lang', type=str, default='bert', help='The language backbone model.')
    parser.add_argument('--lang-layers', type=str, default='-1', help='The language backbone model.')
    parser.add_argument('--dim', type=int, default=64, help='The output dim of the joint emb.')

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--lang-finetune', action='store_true', help='finetune the language encoder.')
    parser.add_argument('--visn-finetune', action='store_true', help='finetune the visual encoder.')
    parser.add_argument('--lang-pretrained', action='store_true', help='Use the pre-trained language encoder.')
    parser.add_argument('--visn-pretrained', action='store_true', help='Use the pre-trained visual encoder.')

    # Optimization
    parser.add_argument("--margin", default=0.5, type=float, help='The margin in the hinge losses.')
    parser.add_argument("--loss", dest='loss', default='paired_hinge',
                        type=str)

    # Training configuration
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument('--output', type=str, default='snap/test')

    # Distributed Training Configuration
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


# args = parse_args()
