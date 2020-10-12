import collections
import os
import pickle
import sys

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
import tqdm
from transformers import BertTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xmatching.data import ImgSentDataset, ImgSentTorchDataset
from xmatching.loss import paired_hinge_rank_loss
from xmatching.metric import batchwise_accuracy, batchwise_recall
from xmatching.model import LangModel, VisnModel, JointModel, LANG_MODELS
from xmatching.param import parse_args


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    args = parse_args()
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus * args.nodes
    # mp.spawn(setup, nprocs=args.gpus, args=(args,))
    # args.world_size = args.gpus * args.nodes
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    device = torch.device('cuda', gpu)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    # Models
    lang_layers = list(map(lambda x: -int(x), args.lang_layers.split(',')))     # The layers concated as the output.
    lang_model = LangModel(args.dim, arch=args.lang, layers=lang_layers,
                           pretrained=args.lang_pretrained, finetuning=args.lang_finetune)
    visn_model = VisnModel(args.dim, arch=args.visn,
                           pretrained=args.visn_pretrained, finetuning=args.visn_finetune)
    # The use of joint model would help synchronization in distributed learning.
    model = JointModel(lang_model, visn_model)

    # Since we will disallow the broadcast of buffers in DDP
    # we want make sure that there are no buffers besides batch normalization and position id.
    for name, buffer in model.named_buffers():
        assert 'bn' in name or 'downsample' in name or "position_ids" in name

    if args.load is not None:
        state_dict = torch.load(args.load, map_location=device)
        new_state_dict = {}
        for key, value in state_dict.items():        # If the ddp state_dict is saved
            if 'num_batches_tracked' not in key:
                if key.startswith("module."):
                    new_state_dict[key[len("module."):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
        model_keys = set(model.state_dict().keys())
        load_keys = set(new_state_dict.keys())
        print("Keys in model but not in load:")
        for key in sorted(model_keys - load_keys):
            print(key)
        print("Keys in load but not in model:")
        for key in sorted(load_keys - model_keys):
            print(key)
        model.load_state_dict(new_state_dict)

    # Pre-processing Hyper-Params
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    Model, Tokenizer, weight = LANG_MODELS[args.lang]
    tokenizer = Tokenizer.from_pretrained(weight)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = args.max_len

    # Dump the pre-processing objs for future feature extractions.
    if gpu == 0:
        pickle.dump(tokenizer, open(
            os.path.join(args.output, 'tokenizer.pkl'), 'wb'))
        pickle.dump(valid_transform, open(
            os.path.join(args.output, 'img_transform.pkl'), 'wb'))

    # Data Sets
    train_set = ImgSentDataset(args.train_imgs, args.train_langs, tiny=args.tiny, fast=args.fast)
    train_tset = ImgSentTorchDataset(
        train_set, train_transform, tokenizer, max_len
    )
    print("GPU %d: load %d data in training." % (gpu, len(train_set)))
    valid_set = ImgSentDataset(args.valid_imgs, args.valid_langs, tiny=args.tiny, fast=args.fast)
    valid_set.shuffle()         # Valid set only gets shuffled once!!!
    print("GPU %d: load %d data in validation." % (gpu, len(valid_set)))
    valid_tset = ImgSentTorchDataset(
        valid_set, valid_transform, tokenizer, max_len
    )
    print()

    # Data Loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_tset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_tset,
        batch_size=(args.batch_size // args.world_size),
        shuffle=False,          # Will be shuffled in the sampler.
        num_workers=max(args.num_workers // args.world_size, 1),
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_tset,
        batch_size=256,             # Fix batch_size to have stable batchwise evaluations.
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if args.optim == 'bert':
        from transformers import AdamW, get_linear_schedule_with_warmup
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        t_total = len(train_loader) * args.epochs
        warmup_steps = int(t_total * args.warmup_ratio)
        print("Train for %d steps and warm up for %d steps" % (t_total, warmup_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )
    else:
        if args.optim == 'sgd':
            optimizer = args.optimizer(
                [param for param in model.parameters() if param.requires_grad],
                args.lr,
                momentum=0.9
            )
        else:
            optimizer = args.optimizer(
                [param for param in model.parameters() if param.requires_grad],
                args.lr,
                # momentum=0.9
            )

    # Loss and optimizer
    criterion = paired_hinge_rank_loss
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    if args.fp16:
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2')
            # Defautly, current apex DDP would not broadcast the buffers.
            model = DDP(model)
        except Exception as e:
            print(e)
            print("Please install apex library")
            return
    else:
        # Note that we disallow broad cast buffers here to reduce communication cost.
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    if args.test_only or args.load:     # Test the loading performance
        if gpu == 0:
            print("Test: GPU %d will test %d data in %d iterations." %
                  (gpu, len(valid_loader) * 256, len(valid_loader)))
            results = valid(args, model, criterion, valid_loader)
            print("Initial test results:")
            for key, value in results.items():
                print('\t%s: %0.4f' % (key, value))
        if args.test_only:
            exit()

    best_valid_loss = 9595.
    for epoch in range(args.epochs):
        if gpu == 0:
            print("Training of Epoch %d: GPU %d will process %d data in %d iterations." %
                  (epoch, gpu, len(train_loader) * args.batch_size // args.world_size, len(train_loader)))
        prev_loss = total_loss = 0.
        for i, (uid, lang_input, visn_input) in enumerate(tqdm.tqdm(train_loader, disable=(gpu!=0))):
            # Currently, lang_input is the (input_ids, attention_mask)
            # visn_input is (tensor_img)
            lang_input = tuple(x.cuda(non_blocking=True) for x in lang_input)
            visn_input = tuple(x.cuda(non_blocking=True) for x in visn_input)

            # Forward pass
            model.zero_grad()
            lang_output, visn_output = model(lang_input, visn_input)
            loss = criterion(lang_output, visn_output, lang_input[1], args.margin)
            total_loss += loss.item()

            # Backward
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Step
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            if args.optim == 'bert':
                scheduler.step()

            # # Logging
            # interval = 100
            # if (i+1) % interval == 0:
            #     print("GPU %d Epoch %d Iter %d: Training Loss %0.4f" %
            #           (gpu, epoch, i+1, (total_loss - prev_loss) / interval))
            #     prev_loss = total_loss
        if gpu == 0:
            print("GPU %d Epoch %d: Total Training Loss %0.4f" % (gpu, epoch, total_loss / len(train_loader)))
            print()
            print("Validation: GPU %d will process %d data in %d iterations." %
                  (gpu, len(valid_loader) * 256, len(valid_loader)))
            results = valid(args, model, criterion, valid_loader, use_tqdm=True)
            for key, value in results.items():
                print('\t%s: %0.4f' % (key, value))
            if results['loss'] < best_valid_loss:
                best_valid_loss = results['loss']
                snap_path = os.path.join(args.output, 'BEST.pth')
                print("GPU 0: Save snapshot to ", snap_path)
                torch.save(model.module.state_dict(), snap_path)
                torch.save(model.module, snap_path + '.model')
            print("BEST valid loss %0.4f" % best_valid_loss)
            print()


def valid(args, model, criterion, valid_loader, use_tqdm=True):
    model.eval()
    results = collections.defaultdict(lambda: 0)
    iterator = tqdm.tqdm(valid_loader) if use_tqdm else valid_loader
    for i, (uid, lang_input, visn_input) in enumerate(iterator):
        # Currently, lang_input is the (input_ids, attention_mask)
        # visn_input is (tensor_img)
        lang_input = tuple(x.cuda(non_blocking=True) for x in lang_input)
        visn_input = tuple(x.cuda(non_blocking=True) for x in visn_input)

        with torch.no_grad():
            # Forward pass
            lang_output, visn_output = model(lang_input, visn_input)

            # Evaluation
            results['loss'] += criterion(lang_output, visn_output, lang_input[1], args.margin).item()
            recall_results = batchwise_recall(lang_output, visn_output, lang_input[1], recalls=(1, 5, 10))
            for key, value in recall_results.items():
                results['R%d' % key] += value

    for key in results:
        results[key] = results[key] / len(valid_loader)
    model.train()

    return results


if __name__ == "__main__":
    main()
