# coding=utf-8
# Copyleft 2020 project COL.

import argparse
from pathlib import Path

from transformers import AutoTokenizer
import time

from to_hdf5 import to_hdf5

def tokenize_dataset(data_dir, fname, tokenizer_name, lines_are_sents=False):
    data_path = Path(data_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    f = open(data_path / fname)
    g = open((data_path / ('%s.%s' % (fname, tokenizer_name))), 'w')

    # Statistics
    dcmt_cnt = 0
    token_cnt = 0
    line_cnt = 0
    line_starts = []

    # Logging and dumping hyper-parameters
    cache = ''
    log_interval = log_iter = 1000000
    dump_interval = dump_iter = 100000
    start_time = time.time()

    for i, line in enumerate(f):
        # Identify the start of documents, ignore it.
        if 'wiki103' in data_dir:
            if line.startswith(' = '):
                dcmt_cnt += 1
                continue
        elif 'wiki' in data_dir:
            if len(line.strip().split(' ')) == 1:
                dcmt_cnt += 1
                continue

        if 'wiki' in data_dir:
            # Remove too short lines. Book corpus does not need this.
            if len(line.strip().split(' ')) < 5:
                continue

        # Drop empty line (1)
        if len(line.strip()) == 0:
            continue

        tokenized_line = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
        # tokenized_line = tokenizer.encode(line, add_special_tokens=False)
        if len(tokenized_line) == 0:    # Drop empty line (2)
            continue

        line_cnt += 1
        line_starts.append(token_cnt)
        if i < 5:
            print()
            print('Line:', line)
            print('Tokens:', ' '.join(tokenizer.convert_ids_to_tokens(tokenized_line)))
        token_cnt += len(tokenized_line)
        cache += ' '.join(map(str, tokenized_line)) + '\n'

        if (token_cnt + 1) > dump_iter:
            g.write(cache)
            cache = ''
            dump_iter += dump_interval

        if (token_cnt + 1) > log_iter:
            used_time = time.time() - start_time
            print("Process %d tokens in %d seconds, %0.4f tokens per second." % (
                token_cnt, used_time, token_cnt / used_time))
            log_iter += log_interval

    # Deal with the last remaining tokens.
    line_starts.append(token_cnt)
    g.write(cache)

    # Dump Line starts
    identifier = 'sent' if lines_are_sents else 'line'
    with open(data_path / ('%s.%s.%s' % (fname, tokenizer_name, identifier)), 'w') as f:
        for line_start in line_starts:
            f.write(str(line_start) + "\n")

    f.close()
    g.close()
    print(f"Documents: {dcmt_cnt}, Lines: {line_cnt}, Words: {token_cnt} in dataset {fname}")

    to_hdf5(str(data_path / fname), tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "datadir", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "fname", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "tokenizer_name", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--lines-are-sents", action='store_true',
        help="Add this if the line are already segmented to sentences, instead of paragraphs."
    )

    param = parser.parse_args()

    tokenize_dataset(
        param.datadir,
        param.fname,
        param.tokenizer_name,
        param.lines_are_sents,
    )

