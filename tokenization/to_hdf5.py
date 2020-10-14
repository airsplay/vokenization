import h5py
import numpy as np
import tqdm

from transformers import AutoTokenizer


def validate_hdf5(fname, tokenizer_name):
    print("--------------------------------------------")
    print("Start to valid the hdf5 file", fname + '.' + tokenizer_name + '.hdf5')

    with open(fname) as f:
        lines = []
        for line in f:
            if 'wiki' in fname:
                # Wiki103: remove document title
                if line.startswith(' = '):
                    continue
                # Full Wiki: Remove the too short lines.
                if len(line.strip().split(' ')) < 5:
                    continue

            if len(line.strip()) == 0:
                # Always drop empty line
                continue
            lines.append(line)

    # Use the slow tokenizer to validate the results of the fast tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    h5_file = h5py.File(fname + '.' + tokenizer_name + '.hdf5', 'r')
    tokens = h5_file['tokens']

    print("Start to check the first 10 lines:")
    ids = []
    for line in lines[:10]:
        ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)))
    ids = np.array(ids)
    first_tokens = np.array(tokens[:len(ids)])
    if np.array_equal(ids, first_tokens):
        print("PASS")
    else:
        print(' '.join(tokenizer.convert_ids_to_tokens(ids)))
        print()
        print(' '.join(tokenizer.convert_ids_to_tokens(first_tokens)))
        assert False, "FAIL"

    print("Start to check the last 10 lines:")
    ids = []
    for line in lines[-10:]:
        ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)))
    ids = np.array(ids)
    last_tokens = np.array(tokens[-len(ids):])
    if np.array_equal(ids, last_tokens):
        print("PASS")
    else:
        print(' '.join(tokenizer.convert_ids_to_tokens(ids)))
        print(' '.join(tokenizer.convert_ids_to_tokens(last_tokens)))
        assert False, "FAIL"
    print("--------------------------------------------")


def to_hdf5(fname, tokenizer_name, validate=True):
    print("Process %s" % fname)

    h5_file = h5py.File(fname + '.' + tokenizer_name + '.hdf5', 'w')
    dset = h5_file.create_dataset("tokens",
                                  (0,),
                                  maxshape=(None,),
                                  dtype='int32')

    dump_interval = 1000000
    dump_iter = 0
    with open('%s.%s' % (fname, tokenizer_name)) as f:
        lines = 0
        tokens = []
        for line in tqdm.tqdm(f):
            for token in map(int, line.split(' ')):
                tokens.append(token)
            if len(tokens) >= dump_interval:
                dset.resize((dump_iter + len(tokens),))
                dset[dump_iter: dump_iter + len(tokens)] = tokens
                dump_iter += len(tokens)
                tokens = []
            lines += 1

        dset.resize((dump_iter + len(tokens),))
        dset[dump_iter: dump_iter + len(tokens)] = tokens
        dump_iter += len(tokens)

    assert len(dset) == dump_iter
    h5_file.close()

    if validate:
        validate_hdf5(fname, tokenizer_name)

    print()

