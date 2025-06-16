import sys
sys.path.append('.')

import os
import json
import glob

import numpy as np
import nibabel as nib

from tqdm import tqdm
from scipy.ndimage import zoom
from joblib import Parallel, delayed

from preprocessing.organ_labels_v2_volumetric import organ_labels

DATA_SPLIT_FILE = '/home/eytan/storage/staff/eytankats/data/nako_10k/nako_dataset_split.json'
LABELS_FILE = '/home/eytan/storage/staff/eytankats/data/nako_10k/labels_aggregated_v2.json'
MASKS_PATTERN = '/home/eytan/storage/staff/eytankats/data/nako_10k/masks_aggregated_v2/*.nii.gz'
OUTPUT_MASKS_DIR = '/home/eytan/storage/staff/eytankats/data/nako_10k/masks_volumetric_v2'
DATA_PARTITION = 'training'


def process(idx, multi_lbl_msk, lbl, available_lbls):
    organ_label_idx = available_lbls[lbl]

    # get mask of single organ
    organ_mask = np.zeros_like(multi_lbl_msk)
    organ_mask[multilabel_mask == organ_label_idx] = 1

    organ_mask = organ_mask.astype(np.uint8)

    # reshape the mask
    target_shape = (256, 64, 256)
    zoom_factors = [t / s for t, s in zip(target_shape, organ_mask.shape)]

    # Resize with linear interpolation
    organ_mask = zoom(organ_mask, zoom=zoom_factors, order=0)

    # get output mask directory and path
    mask_output_dir = os.path.join(OUTPUT_MASKS_DIR, DATA_PARTITION, lbl)
    os.makedirs(mask_output_dir, exist_ok=True)

    mask_output_path = os.path.join(mask_output_dir, idx + '.npy')

    # save mask
    np.save(mask_output_path, organ_mask)


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

    Parallel(n_jobs=10)(delayed(process)(mask_id, multilabel_mask, lbl, available_labels) for lbl in organ_labels)