# In this file, we extract the vision features as the keys in retrieval.
import argparse
import os
import pickle
import shutil
import sys

import h5py
import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import tqdm
from transformers import BertTokenizer
from PIL import Image

import common

# Load all images
Image.MAX_IMAGE_PIXELS = None


def get_img_path(img_set, img_id):
    """
    Get the paths regarding the img_set and img_id.
    THIS FUNCTION MIGHT NEED TO BE MODIFIED.
    """
    source, tag = img_set.split('_')
    if source == 'cc':
        split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (common.CC_ROOT, split_tag, img_id)
    elif 'COCO' in img_id:
        _, split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (common.COCO_ROOT, split_tag, img_id + '.jpg')
    else:   # VG images
        return "%s/images/%s.jpg" % (common.VG_ROOT, img_id)


def get_img_paths_and_ids(img_set):
    """
    Return a list of images paths and image ids in this 'img_set'.
    """

    # Load the image ids from the common local dir,
    # thus make sure that the order of the images are the same.
    info_dir = os.path.join(common.LOCAL_DIR, 'images')
    img_paths = []
    with open(os.path.join(info_dir, img_set + '.ids')) as f:
        img_ids = list(map(lambda x: x.strip(), f.readlines()))
    for img_id in img_ids:
        img_paths.append(get_img_path(img_set, img_id))
    return img_paths, img_ids


def save_img_paths_and_ids(img_set, img_paths, img_ids, output):
    info_dir = os.path.join(common.LOCAL_DIR, 'images')

    # Save Image Paths
    curr_paths_fname = os.path.join(output, img_set + '.path')
    print("\tSave img paths to ", curr_paths_fname)
    with open(curr_paths_fname, 'w') as f:
        for path in img_paths:
            f.write(path + "\n")

    # Save Image Ids
    curr_ids_fname = os.path.join(output, img_set + '.ids')
    print("\tSave img ids to ", curr_ids_fname)
    with open(curr_ids_fname, 'w') as f:
        for idx in img_ids:
            f.write(idx + "\n")

    common_paths_fname = os.path.join(info_dir, img_set + '.path')
    if os.path.exists(common_paths_fname):
        with open(common_paths_fname) as f:
            common_img_paths = f.readlines()
            common_img_paths = [img_path.strip() for img_path in common_img_paths]
            # All feature extractor should extract for the same image set.
            assert common_img_paths == img_paths
    else:
        shutil.copy(curr_paths_fname, common_paths_fname)


def extract_vision_feature_keys(model, img_transform, img_sets, output, batch_size):
    """

    :param model: The visn_model which takes an image [b, channel, H, W] as input,
                  and output with [b, f]
    :param img_transform: The transformation of images, compatible with training.
    :param img_sets: The sets of images to be extracted.
    :param output: The directory to save the extracted keys.
    :return:
    """
    last_dim = -1
    for img_set in img_sets:
        print("Extracting feature keys for image set %s" % img_set)
        img_paths, img_ids = get_img_paths_and_ids(img_set)
        saved_img_paths = []
        saved_img_ids = []
        img_keys = []
        tensor_imgs = []
        for i, img_path in enumerate(tqdm.tqdm(img_paths)):
            try:
                pil_img = default_loader(img_path)
            except Exception as e:
                print(e)
                print("Skip image %s" % img_path)
                continue
            saved_img_paths.append(img_path)
            saved_img_ids.append(img_ids[i])

            tensor_imgs.append(img_transform(pil_img))

            if len(tensor_imgs) == batch_size:
                visn_input = torch.stack(tensor_imgs).cuda()
                with torch.no_grad():
                    visn_output = model(visn_input)

                # Check sizes of features are equal.
                if last_dim == -1:
                    last_dim = visn_output.shape[-1]
                assert last_dim == visn_output.shape[-1]
                last_dim = visn_output.shape[-1]

                # Saved the features in hdf5
                img_keys.extend(visn_output.detach().cpu().numpy())

                tensor_imgs = []

        if len(tensor_imgs) > 0:
            visn_input = torch.stack(tensor_imgs).cuda()
            with torch.no_grad():
                visn_output = model(visn_input)
            # Saved the features in hdf5
            img_keys.extend(visn_output.detach().cpu().numpy())

        assert len(img_keys) == len(saved_img_paths)
        h5_path = os.path.join(output, img_set + '.hdf5')
        print(f"\tSave features (keys) to {h5_path} with hdf5 dataset 'Keys'.")
        h5_file = h5py.File(h5_path, 'w')
        dset = h5_file.create_dataset("keys", (len(saved_img_paths), last_dim))
        for i, img_key in enumerate(img_keys):
            dset[i] = img_key
        save_img_paths_and_ids(img_set, saved_img_paths, saved_img_ids, output)
        h5_file.close()


# This default transformation is used by PyTorch ResNet on ImageNet.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
default_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


import torch
from torch import nn
import torchvision.models as models
def get_visn_arch(arch):
    try:
        return getattr(models, arch)
    except AttributeError as e:
        print(e)
        print("There is no arch %s in torchvision." % arch)

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']


class VisnModel(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True):
        """
        :param dim: dimension of the output
        :param arch: backbone architecture,
        :param pretrained: load feature with pre-trained vector
        :param finetuning: finetune the model
        """
        super().__init__()
        # Setup Backbone
        resnet = get_visn_arch(arch)(pretrained=pretrained)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, img):
        """
        :param img: a tensor of shape [batch_size, H, W, C]
        :return: a tensor of [batch_size, d]
        """
        x = self.backbone(img)
        x = x.detach()
        # x = x / x.norm(2, dim=-1, keepdim=True)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, default=None,
                        help='The directory saved the model (containing'
                             'BEST.pth.model).')
    parser.add_argument('--torchvision-model', type=str, default=None,
                        help='The directory saved the model (containing'
                             'BEST.pth.model).')
    parser.add_argument('--image-sets', type=str, default='coco_minival',
                        help='The splits of images to be extracted')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='The directory to save the extracted feature keys')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    img_sets = [img_set.strip() for img_set in args.image_sets.split(',')]

    if args.torchvision_model is not None:
        assert args.load_dir is None, ("either load from torch model using option 'torchvision_model'"
                                       "or from pre-trained CoX model with option 'load_dir'")
        visn_model = VisnModel(arch=args.torchvision_model).eval().cuda()
        if args.batch_size > 1:
            # for multi-batch extraction, must use the same image size
            img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # For single-batch extraction, we want to extract high-quality features, with two processes:
            #    1. Use large image sizes (400 - 600)
            #    2. Keep the aspect ratio
            MIN_SIZE = 400.
            MAX_SIZE = 600.
            def img_transform_func(img):
                img_w, img_h = img.size     # PiL Image's size order is w, h
                assert img_w > 0 and img_h > 0
                scale = min(
                    MIN_SIZE / min(img_w, img_h),
                    MAX_SIZE / max(img_w, img_h),
                )
                # Keep the aspect ratio
                want_w, want_h = int(img_w * scale), int(img_h * scale)

                _img_transform = transforms.Compose([
                    transforms.Resize((want_h, want_w)),    # PyTorch use size order h, w
                    transforms.ToTensor(),
                    normalize
                ])
                return _img_transform(img)
            img_transform = img_transform_func
    else:
        # Load the model
        if os.path.exists(args.load_dir + '/BEST.pth.model'):
            print("Load model from %s." % (args.load_dir + '/BEST.pth.model'))
            sys.path.append(args.load_dir + '/src')
            for dirc in os.listdir(args.load_dir + '/src'):
                sys.path.append(args.load_dir + '/src/' + dirc)
            # import model        # The pickle has some issues... thus must load the library
            joint_model = torch.load(args.load_dir + '/BEST.pth.model')
            joint_model.eval()            # DO NOT FORGET THIS!!!
            visn_model = joint_model.visn_model
        else:
            print(f"No snapshot {args.load_dir + '/BEST.pth.model'}. Exit.")
            exit()

        # Load the img-preprocessing transformation, which used in training CoX model.
        if os.path.exists(args.load_dir + '/img_transform.pkl'):
            print("Load img transformation from %s." % (args.load_dir + '/img_transform.pkl'))
            with open(args.load_dir + '/img_transform.pkl', 'rb') as f:
                img_transform = pickle.load(f)
        else:
            print("Using default image transformatioin")
            img_transform = default_transform

    # Feature output directory
    output_dir = args.output_dir
    if args.output_dir is None:
        output_dir = args.load_dir + '/keys'      # Save the keys with the model dict
    os.makedirs(output_dir, exist_ok=True)

    extract_vision_feature_keys(
        visn_model,
        img_transform,
        img_sets,
        output_dir,
        args.batch_size
    )
