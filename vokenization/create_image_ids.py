import json
import os
from pathlib import Path
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

imgset2lxrtfname = {
    'coco_train': 'mscoco_train.json',
    'coco_nominival': 'mscoco_nominival.json',
    'coco_minival': 'mscoco_minival.json',
    'vg_nococo': 'vgnococo.json',
}

imgset2ccfname = {
    'cc_train': 'training.tsv',
    'cc_valid': 'validation.tsv'
}


def write_ids(img_set, img_ids):
    """
    Write the indexed image ids 'img_ids' for image set 'img_set' to
    the local file.
    """
    info_dir = os.path.join(common.LOCAL_DIR, 'images')
    os.makedirs(info_dir, exist_ok=True)
    print("Write %d image ids for image set %s to %s." % (
        len(img_ids), img_set, os.path.join(info_dir, img_set + '.ids')))
    ids_path = os.path.join(info_dir, img_set + '.ids')
    if os.path.exists(ids_path):
        # If there is an existing ids_path, make sure that they are the same.
        print(f"Already exist the image ids for image set {img_set} at path {ids_path}.")
        print("Now, we want to make sure that they are equal:")
        with open(ids_path, 'r') as f:
            exist_img_ids = list(map(lambda x: x.strip(), f.readlines()))
        success = True
        for i, (exist_img_id, img_id) in zip(exist_img_ids):
            if exist_img_id != img_id:
                print(f"The image id at line {i} is different:")
                print(f"\tIn the file: {exist_img_id}, In this script: {img_id}")
                success = False
        if success:
            print("PASS!")
    else:
        with open(ids_path, 'w') as f:
            for img_id in img_ids:
                f.write(img_id + '\n')


for img_set in common.IMAGE_SETS:
    if img_set in imgset2lxrtfname:
        lxrt_path = Path(common.LXRT_ROOT)
        img_ids = []
        fname = imgset2lxrtfname[img_set]
        for datum in json.load((lxrt_path / fname).open()):
            img_id = datum['img_id']
            img_ids.append(img_id)

        write_ids(img_set, img_ids)

    if img_set in imgset2ccfname:
        cc_path = Path(common.CC_ROOT)
        img_ids = []
        fname = imgset2ccfname[img_set]
        if not (cc_path / fname).exists():
            print("No such file", cc_path / fname)
            continue
        for i, line in enumerate((cc_path / fname).open()):
            sent, img_id = line.split('\t')
            img_ids.append(img_id.strip())

        write_ids(img_set, img_ids)
