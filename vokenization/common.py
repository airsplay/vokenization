import os

# Name of image sets
IMAGE_SETS = [
    'coco_train',
    'coco_nominival',
    'coco_minival',
    'vg_nococo',
    'cc_train',
    'cc_valid',
]

# Root of each dataset
# CC_ROOT, COCO_ROOT, VG_ROOT should contain the `images` folder
# CC_ROOT -- images
#              |-- training
#                      |-- training_00009486    # Jpeg files but does not have the extension.
#                      |-- ....
#              |-- validation
#                      |-- validation_00009486
#                      |-- ...
# CC_ROOT = os.getenv('CC_ROOT', 'data/cc')
# COCO_ROOT = os.getenv('COCO_ROOT', 'data/mscoco')
# VG_ROOT = os.getenv('VG_ROOT', 'data/vg')
# LXRT_ROOT = os.getenv('LXRT_ROOT', 'data/lxrt')
CC_ROOT = 'data/cc'
COCO_ROOT = 'data/mscoco'
VG_ROOT = 'data/vg'
LXRT_ROOT = 'data/lxmert'

# THe local directory to save essential image infos
#       (e.g., image ids for the vokenizer, image paths in this server)
# LOCAL_DIR
#   |- images
#         |- coco_train_ids.txt
#         |- coco_train_paths.txt
#         |- cc_train_ids.txt
#         |- cc_train_paths.txt
#         |- ..............
# Running create_image_ids.py will build *_ids.txt
# Running extract_vision_keys.py will build *_paths.txt
LOCAL_DIR = 'data/vokenization'

