import sys
sys.path.append('.')

import os
import json
import glob

import numpy as np
import nibabel as nib

from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

from preprocessing.organ_labels import organ_labels

DATA_SPLIT_FILE = '/home/kats/storage/staff/eytankats/data/nako_10k/nako_dataset_split.json'
LABELS_FILE = '/home/kats/storage/staff/eytankats/data/nako_10k/labels_aggregated.json'
MASKS_PATTERN = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_aggregated/*.nii.gz'
OUTPUT_MASKS_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_projection'
DATA_PARTITION = 'validation'


def process(idx, multi_lbl_msk, lbl, available_lbls):
    organ_label_idx = available_lbls[lbl]

    # get mask of single organ
    organ_mask = np.zeros_like(multi_lbl_msk)
    organ_mask[multilabel_mask == organ_label_idx] = 1

    # get depth image of organ and normalize by maximum depth of the depth body image
    depth_organ_mask = np.argmax(organ_mask, axis=1)
    depth_organ_mask = depth_organ_mask / multi_lbl_msk.shape[1]
    depth_organ_mask[depth_organ_mask > 0] = 1 - depth_organ_mask[depth_organ_mask > 0]

    # get output mask directory and path
    mask_output_dir = os.path.join(OUTPUT_MASKS_DIR, DATA_PARTITION, lbl)
    os.makedirs(mask_output_dir, exist_ok=True)

    mask_output_path = os.path.join(mask_output_dir, idx + '.png')

    # save organ depth image
    im = np.uint8(depth_organ_mask * 255)
    im = Image.fromarray(im)
    im.save(mask_output_path)


os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

masks_paths = sorted(glob.glob(MASKS_PATTERN))
masks_ids = [os.path.basename(mask_path)[:6] for mask_path in masks_paths]
partition_masks_ids = [mask_name[:6] for mask_name in json.load(open(DATA_SPLIT_FILE))[DATA_PARTITION]]

available_labels = json.load(open(LABELS_FILE))["labels"]

for mask_id in tqdm(partition_masks_ids):

    if mask_id in masks_ids:
        mask_path = masks_paths[masks_ids.index(mask_id)]
    else:
        print(f'Mask {mask_id} not found, skipping.')
        continue

    # load multilabel mask
    multilabel_mask = nib.load(mask_path).get_fdata()

    # flip mask
    multilabel_mask = np.flip(multilabel_mask, axis=1)
    multilabel_mask = np.flip(multilabel_mask, axis=0)

    # get maximum height
    max_height = multilabel_mask.shape[1]

    Parallel(n_jobs=10)(delayed(process)(mask_id, multilabel_mask, lbl, available_labels) for lbl in organ_labels)