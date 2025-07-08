import sys
sys.path.append('.')

import os
import glob
import numpy as np
import nibabel as nib

from tqdm import tqdm

# Own Package
from preprocessing.organ_labels_v2_volumetric import organ_labels

# Get data loader
IMAGE_DIR = '/home/eytan/storage/staff/eytankats/data/nako_10k/images_depth'
MASK_DIR = '/home/eytan/materials/orgloc/data/nako_10k/masks_volumetric_preprocessed_v2'
PARTITION = 'training'

# Create id - mask map
masks_paths = glob.glob(os.path.join(MASK_DIR, '*.nii.gz'))
masks_ids = [os.path.basename(mask_path)[:6] for mask_path in masks_paths]
masks_map = dict(zip(masks_ids, masks_paths))

# Get valid ids
image_paths = glob.glob(os.path.join(IMAGE_DIR, PARTITION, '*.png'))
image_ids = [os.path.basename(n)[:6] for n in image_paths]

valid_ids = list(set(image_ids) & set(masks_ids))

# Create placeholders for mean segmenation for selected anatomies
mask = nib.load(masks_map[valid_ids[0]]).get_fdata()
mean_segs = [np.zeros(shape=mask.shape, dtype=np.float32) for _ in organ_labels]

# Iterate other masks
for valid_id in tqdm(valid_ids):

    mask = nib.load(masks_map[valid_id]).get_fdata()

    # Iterate other selected anatomies
    for anatomy_idx, anatomy in enumerate(organ_labels):
        mean_segs[anatomy_idx][mask == anatomy_idx + 1] += 1

# Save mean binary masks for each anatomy
mean_segs = [m/len(valid_ids) for m in mean_segs]

for anatomy_idx, anatomy in enumerate(organ_labels):
    mean_seg_nib = nib.Nifti1Image(mean_segs[anatomy_idx], np.eye(4))
    nib.save(mean_seg_nib, os.path.join(MASK_DIR, 'mean_' + anatomy + '.nii.gz'))






