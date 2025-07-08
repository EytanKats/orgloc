import sys
sys.path.append('.')

import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from tqdm import tqdm
from tabulate import tabulate
from monai.transforms import Resize

# Own Package
from utils.tools import mask_to_bbox_volumetric
from preprocessing.organ_labels_v2_volumetric import selected_organ_labels

# Get data loader
IMAGE_DIR = '/home/eytan/storage/staff/eytankats/data/nako_10k/images_depth'
MASK_DIR = '/home/eytan/materials/orgloc/data/nako_10k/masks_volumetric_preprocessed_v2'
LABELS_FILE = '/home/eytan/storage/staff/eytankats/data/nako_10k/labels_processed_aggregated_v2.json'
OUTPUT_DIR = '/home/eytan/materials/orgloc/experiments/mean_model'
PARTITION = 'test'

# Create id - mask map
masks_paths = glob.glob(os.path.join(MASK_DIR, '*.nii.gz'))
masks_ids = [os.path.basename(mask_path)[:6] for mask_path in masks_paths]
masks_map = dict(zip(masks_ids, masks_paths))

# Get valid ids
image_paths = glob.glob(os.path.join(IMAGE_DIR, PARTITION, '*.png'))
image_ids = [os.path.basename(n)[:6] for n in image_paths]

valid_ids = list(set(image_ids) & set(masks_ids))

# Load and reorient mean masks
print('Loading mean segmentation models...')
mean_segs = [nib.load(os.path.join(MASK_DIR, 'mean_' + anatomy + '.nii.gz')).get_fdata() for anatomy in selected_organ_labels]
mean_segs = [np.flip(mean_seg, axis=1) for mean_seg in mean_segs]

# Resize mean masks to 1x1x1mm resolution
print('Resizing mean segmentation models...')
mean_seg_resize_transform = Resize(spatial_size=(390, 480, 948))
mean_segs = [mean_seg_resize_transform(torch.tensor(mean_seg.copy()).unsqueeze(0)).squeeze().numpy() for mean_seg in mean_segs]

# Iterate other masks
available_labels = json.load(open(LABELS_FILE))["labels"]
mask_resize_transform = Resize(spatial_size=(390, 480, 948), mode='nearest')

name_list = []

left = {}
right = {}
superior = {}
inferior = {}
anterior = {}
posterior = {}
for anatomy in selected_organ_labels:
    left[anatomy] = []
    right[anatomy] = []
    superior[anatomy] = []
    inferior[anatomy] = []
    anterior[anatomy] = []
    posterior[anatomy] = []
left['mean'] = []
right['mean'] = []
superior['mean'] = []
inferior['mean'] = []
anterior['mean'] = []
posterior['mean'] = []

print('Iterate other test cases and calculate metrics...')
for valid_id in tqdm(valid_ids[:10]):

    name_list.append(valid_id)

    multilabel_mask = nib.load(masks_map[valid_id]).get_fdata()
    multilabel_mask = np.flip(multilabel_mask, axis=1)

    # Resize mask to 1x1x1mm resolution
    multilabel_mask = mask_resize_transform(torch.tensor(multilabel_mask.copy()).unsqueeze(0)).squeeze().numpy()

    # Choose selected labels
    selected_labels_mask = np.zeros_like(multilabel_mask)
    for anatomy_idx, anatomy in enumerate(selected_organ_labels):
        organ_label_idx = available_labels[anatomy]
        selected_labels_mask[multilabel_mask == organ_label_idx] = anatomy_idx + 1

    # Iterate other selected anatomies
    left_mean = 0
    right_mean = 0
    superior_mean = 0
    inferior_mean = 0
    anterior_mean = 0
    posterior_mean = 0
    for anatomy_idx, anatomy in enumerate(selected_organ_labels):

        seg_anatomy = np.zeros_like(selected_labels_mask)
        seg_anatomy[selected_labels_mask == anatomy_idx + 1] = 1

        bbox_seg = mask_to_bbox_volumetric(seg_anatomy)
        bbox_pred = mask_to_bbox_volumetric(mean_segs[anatomy_idx] > mean_segs[anatomy_idx].max() / 2)

        if bbox_seg is not None and bbox_pred is not None:
            inferior_anatomy = bbox_seg['x1'] - bbox_pred['x1']
            right_anatomy = bbox_seg['y1'] - bbox_pred['y1']
            superior_anatomy = bbox_pred['x2'] - bbox_seg['x2']
            left_anatomy = bbox_pred['y2'] - bbox_seg['y2']
            anterior_anatomy = bbox_seg['z1'] - bbox_pred['z1']
            posterior_anatomy = bbox_pred['z2'] - bbox_seg['z2']
        else:
            left_anatomy = np.nan
            right_anatomy = np.nan
            superior_anatomy = np.nan
            inferior_anatomy = np.nan
            anterior_anatomy = np.nan
            posterior_anatomy = np.nan

        left[anatomy].append(left_anatomy)
        left_mean += left_anatomy

        right[anatomy].append(right_anatomy)
        right_mean += right_anatomy

        superior[anatomy].append(superior_anatomy)
        superior_mean += superior_anatomy

        inferior[anatomy].append(inferior_anatomy)
        inferior_mean += inferior_anatomy

        anterior[anatomy].append(anterior_anatomy)
        anterior_mean += anterior_anatomy

        posterior[anatomy].append(posterior_anatomy)
        posterior_mean += posterior_anatomy

    left['mean'].append(left_mean / len(selected_organ_labels))
    right['mean'].append(right_mean / len(selected_organ_labels))
    superior['mean'].append(superior_mean / len(selected_organ_labels))
    inferior['mean'].append(inferior_mean / len(selected_organ_labels))
    anterior['mean'].append(anterior_mean / len(selected_organ_labels))
    posterior['mean'].append(posterior_mean / len(selected_organ_labels))

# MEAN & Std Value
name_list.extend(['Avg', 'Std'])
left['mean'].extend([np.nanmean(left['mean']), np.nanstd(left['mean'], ddof=1)])
right['mean'].extend([np.nanmean(right['mean']), np.nanstd(right['mean'], ddof=1)])
superior['mean'].extend([np.nanmean(superior['mean']), np.nanstd(superior['mean'], ddof=1)])
inferior['mean'].extend([np.nanmean(inferior['mean']), np.nanstd(inferior['mean'], ddof=1)])
anterior['mean'].extend([np.nanmean(anterior['mean']), np.nanstd(anterior['mean'], ddof=1)])
posterior['mean'].extend([np.nanmean(posterior['mean']), np.nanstd(posterior['mean'], ddof=1)])

for anatomy_idx, anatomy in enumerate(selected_organ_labels):
    left[anatomy].extend([np.nanmean(left[anatomy]), np.nanstd(left[anatomy], ddof=1)])
    right[anatomy].extend([np.nanmean(right[anatomy]), np.nanstd(right[anatomy], ddof=1)])
    superior[anatomy].extend([np.nanmean(superior[anatomy]), np.nanstd(superior[anatomy], ddof=1)])
    inferior[anatomy].extend([np.nanmean(inferior[anatomy]), np.nanstd(inferior[anatomy], ddof=1)])
    anterior[anatomy].extend([np.nanmean(anterior[anatomy]), np.nanstd(anterior[anatomy], ddof=1)])
    posterior[anatomy].extend([np.nanmean(posterior[anatomy]), np.nanstd(posterior[anatomy], ddof=1)])

# save csv
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, 'results.csv')

df = pd.DataFrame({
    'name': name_list,
    'left_mean':  left['mean'],
    'right_mean': right['mean'],
    'superior_mean': superior['mean'],
    'inferior_mean': inferior['mean'],
    'anterior_mean': anterior['mean'],
    'posterior_mean': posterior['mean']
})

for anatomy in selected_organ_labels:
    df['left_' + anatomy] = left[anatomy]
    df['right_' + anatomy] = right[anatomy]
    df['superior_' + anatomy] = superior[anatomy]
    df['inferior_' + anatomy] = inferior[anatomy]
    df['anterior_' + anatomy] = anterior[anatomy]
    df['posterior_' + anatomy] = posterior[anatomy]

df.to_csv(csv_path, index=False)

print(tabulate(df.tail(2), headers='keys', tablefmt='psql'))









