# coding=utf-8
# Copyleft 2020 project COL.

import argparse
import copy
from multiprocessing import Queue, Process
import os
import queue
import sys
import time

import h5py
import torch
import tqdm
from spacy.lang.en import English

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vokenization.vokenization import load_model_and_tokenizer, Vokenizer
from vokenization.revokenization import ReVokenizer


# Handle the GPU issue in multi-processing.
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def processer(args, input_queue, output_queue):
    print(f"Setup workers on gpu {args.gpus}")
    img_sets = sorted([img_set.strip() for img_set in args.image_sets.split(',')])

    print("Build models and tokenizer")
    # We will assign the GPU to model latter, thus load to cpu first!
    model, tokenizer = load_model_and_tokenizer(args.load, cpu=True)
    keys_dir = args.load + '/keys'  # Save the keys with the model dict

    print("Build Retriever from %s with image sets" % keys_dir, img_sets)
    vokenizer = Vokenizer(model, tokenizer, keys_dir,
                          img_sets=img_sets, max_img_num=args.max_img_num,
                          gpus=args.gpus, sent_level=('sent' in args.load))
    print(f"GPU: {args.gpus}, build vokenizer with {vokenizer.img_num} images.")

    # Before vokenization, save the image ids
    dset_name = os.path.split(args.corpus)[-1]
    modifier = f".{vokenizer.img_num}" if vokenizer.img_num != 50000 else ""
    vokens_img_ids_path = os.path.join(
        args.output,
        f"{dset_name}.{'_'.join(img_sets)}{modifier}.ids"
    )
    if args.gpus[0] == 0:
        if os.path.exists(vokens_img_ids_path):
            # If the img_ids file exists, assert that they are the same.
            saved_img_ids = open(vokens_img_ids_path).readlines()
            img_ids = vokenizer.img_ids
            assert len(saved_img_ids) == len(img_ids)
            for saved_img_id, img_id in zip(saved_img_ids, img_ids):
                assert saved_img_id.strip() == img_id
        else:
            vokenizer.dump_img_ids(vokens_img_ids_path)

    while True:
        page_id, sents = input_queue.get()
        # Print the first few sents for debugging
        if args.gpus[0] == 0:
            if page_id < 12 and sents is not None:
                print('page_id:', page_id)
                print('batch_size:', len(sents))
                print('ids of sent[0]:', sents[0])
                print('tokens of sent[0]:', tokenizer.convert_ids_to_tokens(sents[0]))
                print()
        # print(f"Processer {args.gpus}: Get Page Id {page_id}")
        if sents is not None:
            output_str = ''
            results = vokenizer.vokenize_ids(sents)
            idxs = results[1]
            for j, idx in enumerate(idxs):
                assert len(idx[1:-1]) == len(sents[j])
                dump_idx = map(lambda x: str(x.item()), idx[1:-1])
                output_str += ' '.join(dump_idx) + '\n'

            output_queue.put((page_id, output_str))
        else:
            break


def reducer(output_fname, output_queue, total_tokens):
    next_page_id = 0
    heap = queue.PriorityQueue()
    output = open(output_fname, 'a')
    cache = ""
    start_time = None
    processed_tokens = 0

    while True:
        page_id, result = output_queue.get()
        if start_time is None:      # The clock starts to tick when receiving the first package.
            start_time = time.time()
        # print("Reducer: Get Page Id %d" % page_id)
        if result is not None:
            # Put it into the heap
            heap.put((page_id, result))

            # Check the could-be-dumped data in the queue
            while heap.qsize() > 0:
                smallest_page_id, result = heap.get()
                if smallest_page_id == next_page_id:
                    # which means that this page is the next page, thus dump it
                    # print("Reducer: Commit Page Id %d" % next_page_id)
                    processed_tokens += len(result.split(' '))
                    cache += result
                    next_page_id += 1
                else:
                    heap.put((smallest_page_id, result))
                    break
            # print("Reducer: Length of Cache Now", len(cache))
            if len(cache) > 1000000:
                # Dump for every 1000000 characters to reduce IO calls
                output.write(cache)
                output.flush()
                cache = ''
                used_time = int(time.time() - start_time)
                print("Process %d tokens, %d to go, with speed %0.2f tokens/second,"
                      "finished in %0.2f hours" % (
                    processed_tokens,
                    total_tokens - processed_tokens,
                    processed_tokens / used_time,
                    (total_tokens - processed_tokens) / (processed_tokens / used_time) / 3600
                ))
        else:
            if len(cache) > 0:
                output.write(cache)
                output.flush()
                cache = ''
            break

    output.close()


def setup_mp(args, tokens, sent_ranges, vokens_path):
    QUEUE_SIZE = 10000
    input_queue = Queue(maxsize=QUEUE_SIZE)
    output_queue = Queue(maxsize=QUEUE_SIZE)

    workers = []
    num_gpu = torch.cuda.device_count()
    for worker_id in range(args.num_workers):
        gpu_id = worker_id % num_gpu
        curr_args = copy.copy(args)
        curr_args.gpus = (gpu_id,)
        worker = Process(target=processer,
                         args=(curr_args, input_queue, output_queue))
        worker.daemon = True
        worker.start()
        workers.append(worker)

    total_tokens = len(tokens) - sent_ranges[0][0] if len(sent_ranges) > 0 else 0
    reduce = Process(target=reducer,
                     args=(vokens_path, output_queue, total_tokens))
    reduce.start()

    for i, start_id in enumerate(range(0, len(sent_ranges), args.batch_size)):
        sents = []
        for left, right in sent_ranges[start_id: start_id + args.batch_size]:
            sents.append(tokens[left: right])
        input_queue.put((i, sents))

    # Notifying workers the end of input
    for _ in workers:
        input_queue.put((-1, None))

    # wait for workers to terminate
    for w in workers:
        w.join()

    # Notify the reducer the end of output
    output_queue.put((-1, None))

    # wait for reducer to terminate
    reduce.join()


def segment_sent(
        tokens,
        tokenizer,
        tokens_line_info_path,
        tokens_sent_info_path
    ):
    """
    Single-processed segmentation of sentences. We might need to parallel this as well.
    """
    with open(tokens_line_info_path) as f:
        line_starts = list(map(int, f.readlines()))

    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    sent_starts = [0]
    now = 0
    for i in tqdm.tqdm(range(len(line_starts) - 1)):
        start_token_idx = line_starts[i]
        end_token_idx = line_starts[i + 1]
        line_tokens = tokens[start_token_idx: end_token_idx]
        line = ' '.join(tokenizer.convert_ids_to_tokens(line_tokens))
        line = line.replace("[UNK]", "UNK")

        doc = nlp(line)
        sents_len = 0
        sents = []
        for sent in doc.sents:
            if i < 2:
                print(sent)
            sent = str(sent)
            sents.append(sent)
            words = sent.split(' ')
            sent_len = len(words)
            now += sent_len
            sent_starts.append(now)
            sents_len += sent_len

        if sents_len != len(line_tokens):
            print(sents_len)
            print(sents)
            print(len(line_tokens))
            print(line)
            assert False
        assert sent_starts[-1] == end_token_idx

    with open(tokens_sent_info_path, 'w') as f:
        for sent_start in sent_starts:
            f.write(str(sent_start) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Text
    parser.add_argument('--corpus', type=str, default='/ssd-playpen/data/wiki103/wiki.train.raw')
    # Models
    parser.add_argument('--load', type=str,
                        default='/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/coco_hinge05_dim64_resxt101_robertal4',
                        help='The directory saved the model (containing'
                             'BEST.pth.model).')
    parser.add_argument('--output', type=str, default=None,
                        help='The directory to save the extracted feature keys.'
                             '"None" would save in the "load" dir')
    parser.add_argument('--backward-tokenizer-name', type=str, default='roberta-base')
    parser.add_argument('--forward-tokenizer-name', type=str, default='roberta-base')
    # Vision: Define the vokens set
    parser.add_argument('--image-sets', type=str, default='vg_nococo',
                        help='The splits of images to be extracted')
    parser.add_argument('--max-img-num', type=int, default=50000,
                        help='number of images used. -1 means all images.')
    # Speed Up Options:
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='-1 will use all GPUs.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='The # of sentences in a batch.')
    args = parser.parse_args()

    if args.num_workers == -1:
        args.num_workers = torch.cuda.device_count()

    if args.output is None:
        args.output = os.path.join(args.load, 'vokens')
    os.makedirs(args.output, exist_ok=True)

    dset_name = os.path.split(args.corpus)[-1]
    img_sets = sorted([img_set.strip() for img_set in args.image_sets.split(',')])
    print()
    print("Main Th"
          "read: Build a virtual vokenizer to check the number of images.")
    keys_dir = args.load + '/keys'  # Save the keys with the model dict
    virtual_vokenizer = Vokenizer(
        None, None, keys_dir,
        img_sets=img_sets, max_img_num=args.max_img_num,
        gpus=(-1,), sent_level=('sent' in args.load))
    modifier = f".{virtual_vokenizer.img_num}" if virtual_vokenizer.img_num != 50000 else ""
    vokens_path = os.path.join(
        args.output,
        f"{dset_name}.{'_'.join(img_sets)}{modifier}"
    )
    tokens_hdf5_path = f'{args.corpus}.{args.backward_tokenizer_name}.hdf5'
    tokens_sent_info_path = f'{args.corpus}.{args.backward_tokenizer_name}.sent'

    # "Load" tokens from hdf5
    tokens_hdf5 = h5py.File(tokens_hdf5_path, 'r')
    tokens = tokens_hdf5['tokens']

    # Calibrate the start line if the vokens have been proceeded.
    if not os.path.exists(tokens_sent_info_path):
        tokens_line_info_path = f'{args.corpus}.{args.backward_tokenizer_name}.line'
        model, tokenizer = load_model_and_tokenizer(args.load, cpu=True)
        segment_sent(
            tokens,
            tokenizer,
            tokens_line_info_path,
            tokens_sent_info_path
        )

    # Load sent info and find the start sentence
    with open(tokens_sent_info_path) as f:
        sent_starts = list(map(int, f.readlines()))

    # Skip the sentences which have been extracted.
    extracted_tokens = 0
    if os.path.isfile(vokens_path):
        with open(vokens_path, 'r') as g:
            for g_line in tqdm.tqdm(g):
                extracted_tokens += len(g_line.strip().split(' '))
    try:
        start_sent_idx = sent_starts.index(extracted_tokens)
    except ValueError as e:
        print("The extracted tokens does not match a starting sentence.")
        print(e)

    # Start to vokenize
    print("Main Thread: Dump visual tokens to %s" % vokens_path)
    print("Main Thread: Start vokenization from the %d'th token" % sent_starts[start_sent_idx])

    sent_ranges = []
    for i in range(start_sent_idx, len(sent_starts) - 1):
        left_token_idx = sent_starts[i]
        right_token_idx = sent_starts[i + 1]
        sent_ranges.append((left_token_idx, right_token_idx))

    setup_mp(args, tokens, sent_ranges, vokens_path)

    # save into hdf5 file
    if os.path.exists(vokens_path + '.hdf5'):
        print("The hdf5 file %s already exists. So the hdf5 file is not converted."
              % (vokens_path + '.hdf5'))
    else:
        with open(args.corpus + '.' + args.backward_tokenizer_name + ".sent") as f:
            for i, line in enumerate(f):
                pass
            num_tokens = int(line)
            num_sents = i

        h5_file = h5py.File(vokens_path + '.hdf5', 'w')
        dset = h5_file.create_dataset("vokens", (num_tokens,), dtype='int32')

        dump_interval = 100000
        dump_iter = 0
        lines = 0

        with open(vokens_path) as f:
            tokens = []
            for line in tqdm.tqdm(f, total=num_sents):
                for token in map(int, line.split(' ')):
                    tokens.append(token)
                if len(tokens) >= dump_interval:
                    dset[dump_iter: dump_iter + len(tokens)] = tokens
                    dump_iter += len(tokens)
                    tokens = []
                lines += 1
            dset[dump_iter: dump_iter + len(tokens)] = tokens
            dump_iter += len(tokens)
            assert num_tokens == dump_iter
            print(lines, num_sents)
            assert lines == num_sents
        h5_file.close()


