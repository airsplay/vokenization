import argparse
import math
import os
from pathlib import Path
from pprint import pprint
import subprocess
import threading
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load", default=None, type=str,
    help="The model loaded, e.g., snap/vlm/wiki103_small"
)
parser.add_argument(
    "--gpus", default=None, type=str,
    help="The list of GPU ids, separated by comma, e.g., '2,3'"
)
parser.add_argument(
    "--snaps", default=1, type=int,
    help="The number of snaps evaluated with GLUE benchmark. "
         "-1 means all."
)
parser.add_argument(
    "--start-from", default=0, type=int
)
args = parser.parse_args()

if args.gpus is None:
    # Get all gpus available in this server.
    num_gpus = torch.cuda.device_count()
    # The device id are labeled from 0 to num_gpus-1.
    available_gpus = list(range(num_gpus))
else:
    available_gpus = [int(gpu_id) for gpu_id in args.gpus.split(",")]
    num_gpus = len(available_gpus)

resource = threading.Semaphore(num_gpus)


def get_snap_paths(load):
    load_path = Path(load)
    paths = []
    for dir_path in load_path.iterdir():
        if dir_path.name.startswith("checkpoint-"):
            paths.append(dir_path)
    return paths


def sorted_paths(paths):
    pathXkey = []
    for path in paths:
        name = path.name
        identifier = name[len("checkpoint-"):]
        if identifier == 'last':
            continue
        if 'epoch' in identifier:
            key = identifier
        else:
            key = int(identifier)
        pathXkey.append((path, key))
    pathXkey = sorted(pathXkey, key=lambda x: x[1])
    paths = list(map(lambda x: x[0], pathXkey))
    return paths


def get_test_paths(paths, snaps):
    """
    Return $snaps paths to be tested on GLUE
    """
    if snaps == -1:
        return paths
    interval = len(paths) * 1. / snaps
    test_paths = []
    for i in range(1, snaps+1):
        idx = int(math.ceil(interval * i)) - 1
        test_paths.append(paths[idx])
    return test_paths


# Get all paths needs to be processed
paths = get_snap_paths(args.load)
paths = sorted_paths(paths)
paths = paths[args.start_from:]
paths = get_test_paths(paths, args.snaps)
paths = paths[::-1]         # Run the last epochs first.
path_lock = threading.Lock()


def run_glue():
    while True:
        # Only have one atomic operation (list.pop) here, do not need lock.
        # A Semaphore is enough to control the resources.
        resource.acquire()
        gpu_id = available_gpus.pop(0)

        # Involve multiple atomic operations (list.__len__, list.pop),
        # thus introduce a lock here.
        path_lock.acquire()
        if len(paths) > 0:
            path = paths.pop(0)
        else:
            path_lock.release()
            break
        path_lock.release()

        model = path.parent
        ckpt = path.name
        print(gpu_id, model, ckpt)
        process = subprocess.Popen(
            ['bash',
             'scripts/run_glue_at_epoch.bash',
             str(gpu_id),    # Use GPU
             '3',            # Number of epochs
             model,
             ckpt
             ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        available_gpus.append(gpu_id)
        resource.release()

        # Sleep here allows the script (run_glue_at_epoch.bash) to finish
        # thus all memory in GPU will be cleared.
        time.sleep(5)
    return


# Allocate #threads which equals to #GPUs
threads = []
for _ in range(num_gpus):
    threads.append(
        threading.Thread(target=run_glue)
    )
for thread in threads:
    thread.start()

# Join to the main thread, thus the main thread will wait for all the threads.
for thread in threads:
    thread.join()
