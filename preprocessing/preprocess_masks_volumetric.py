import sys
sys.path.append('.')

import os
import json
import glob

import numpy as np
import nibabel as nib

from tqdm import tqdm
from scipy import ndimage
from joblib import Parallel, delayed

from preprocessing.organ_labels_v2_volumetric import selected_organ_labels

DATA_SPLIT_FILE = '/home/kats/storage/staff/eytankats/data/nako_10k/nako_dataset_split.json'
LABELS_FILE = '/home/kats/storage/staff/eytankats/data/nako_10k/labels_aggregated_v2.json'
MASKS_PATTERN = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_aggregated_v2/*.nii.gz'
OUTPUT_MASKS_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_volumetric_preprocessed_v2'
DATA_PARTITION = 'validation'


def process(mask_id, masks_ids, available_labels, organ_labels):

    print(f'Processing mask {mask_id}')

    mask_path = masks_paths[masks_ids.index(mask_id)]

    # load multilabel mask
    multilabel_mask = nib.load(mask_path).get_fdata()

    # flip mask
    multilabel_mask = np.flip(multilabel_mask, axis=1)
    multilabel_mask = np.flip(multilabel_mask, axis=0)

    multilabel_mask_proc = np.zeros_like(multilabel_mask, dtype=np.uint8)  # placeholder for the preprocessed file
    for organ_lbl_idx, lbl in enumerate(organ_labels):

        # Get organ mask and define expected number of connecteced components
        if lbl == 'heart':

            heart_lbls = [
                "heart_atrium_left",
                "heart_atrium_right",
                "heart_myocardium",
                "heart_ventricle_left",
                "heart_ventricle_right"
            ]

            organ_mask = np.zeros_like(multilabel_mask)
            for heart_lbl in heart_lbls:
                available_label_idx = available_labels[heart_lbl]
                organ_mask[multilabel_mask == available_label_idx] = 1

            n = 1

        elif lbl == 'lung_left':

            lung_lbls = [
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
            ]

            organ_mask = np.zeros_like(multilabel_mask)
            for lung_lbl in lung_lbls:
                available_label_idx = available_labels[lung_lbl]
                organ_mask[multilabel_mask == available_label_idx] = 1

            n = 1

        elif lbl == 'lung_right':

            lung_lbls = [
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right"
            ]

            organ_mask = np.zeros_like(multilabel_mask)
            for lung_lbl in lung_lbls:
                available_label_idx = available_labels[lung_lbl]
                organ_mask[multilabel_mask == available_label_idx] = 1

            n = 1

        else:
            organ_mask = np.zeros_like(multilabel_mask)
            available_label_idx = available_labels[lbl]
            organ_mask[multilabel_mask == available_label_idx] = 1

            if lbl == 'thyroid_gland':
                n = 2
            else:
                n = 1

        ### Smooth boundaries
        # radius = 10
        # struct = ndimage.generate_binary_structure(rank=3, connectivity=1)  # 6-connectivity
        # for _ in range(radius - 1):
        #     struct = ndimage.binary_dilation(struct)
        #
        # # First remove small protrusions (opening), then fill small holes (closing)
        # smooth = organ_mask.copy().astype(bool)
        # for _ in range(10):
        #     smooth = ndimage.binary_opening(smooth, structure=struct)
        #     smooth = ndimage.binary_closing(smooth, structure=struct)

        ### Fill holes
        organ_mask = ndimage.binary_fill_holes(organ_mask)

        ### Remove small connected components
        organ_mask = remove_small_cc(organ_mask, num_cc=n, mask_id=mask_id, lbl=lbl)

        multilabel_mask_proc[organ_mask == 1] = organ_lbl_idx + 1

    multilabel_mask_proc_nib = nib.Nifti1Image(multilabel_mask_proc, np.eye(4))
    nib.save(multilabel_mask_proc_nib, os.path.join(OUTPUT_MASKS_DIR, mask_id + '.nii.gz'))

def remove_small_cc(mask, num_cc, mask_id, lbl):

    struct = ndimage.generate_binary_structure(rank=3, connectivity=6)
    labels, num = ndimage.label(mask, structure=struct)

    if num == 0 or num_cc >= num:
        return mask

    # Compute voxel counts for each label
    counts = ndimage.sum(np.ones_like(labels, dtype=np.int64), labels, index=np.arange(1, num + 1))
    counts = counts.astype(np.int64)

    print(f'For {mask_id} and {lbl} number of connected components is {num} with size {counts}.')

    # Find labels of the n largest components
    keep = np.argsort(counts)[-num_cc:] + 1  # +1 because labels start at 1

    # Build mask of voxels to preserve
    organ_mask = np.isin(labels, keep)

    return organ_mask

os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

masks_paths = sorted(glob.glob(MASKS_PATTERN))
masks_ids = [os.path.basename(mask_path)[:6] for mask_path in masks_paths]
partition_masks_ids = [mask_name[:6] for mask_name in json.load(open(DATA_SPLIT_FILE))[DATA_PARTITION]]

available_labels = json.load(open(LABELS_FILE))["labels"]
Parallel(n_jobs=10)(delayed(process)(msk_id, masks_ids, available_labels, selected_organ_labels) for msk_id in partition_masks_ids if msk_id in masks_ids)

# for mask_id in tqdm(partition_masks_ids):
#
#     if mask_id in masks_ids:
#         mask_path = masks_paths[masks_ids.index(mask_id)]
#     else:
#         print(f'\nMask {mask_id} not found, skipping.')
#         continue
#
#     # load multilabel mask
#     multilabel_mask = nib.load(mask_path).get_fdata()
#
#     # flip mask
#     multilabel_mask = np.flip(multilabel_mask, axis=1)
#     multilabel_mask = np.flip(multilabel_mask, axis=0)
#
#     multilabel_mask_proc = np.zeros_like(multilabel_mask, dtype=np.uint8)  # placeholder for the preprocessed file
#     for organ_lbl_idx, lbl in tqdm(enumerate(selected_organ_labels)):
#
#         # Get organ mask and define expected number of connecteced components
#         if lbl == 'heart':
#
#             heart_lbls = [
#                 "heart_atrium_left",
#                 "heart_atrium_right",
#                 "heart_myocardium",
#                 "heart_ventricle_left",
#                 "heart_ventricle_right"
#             ]
#
#             organ_mask = np.zeros_like(multilabel_mask)
#             for heart_lbl in heart_lbls:
#                 available_label_idx = available_labels[heart_lbl]
#                 organ_mask[multilabel_mask == available_label_idx] = 1
#
#             n = 1
#
#         elif lbl == 'lung_left':
#
#             lung_lbls = [
#                 "lung_upper_lobe_left",
#                 "lung_lower_lobe_left",
#             ]
#
#             organ_mask = np.zeros_like(multilabel_mask)
#             for lung_lbl in lung_lbls:
#                 available_label_idx = available_labels[lung_lbl]
#                 organ_mask[multilabel_mask == available_label_idx] = 1
#
#             n = 1
#
#         elif lbl == 'lung_right':
#
#             lung_lbls = [
#                 "lung_upper_lobe_right",
#                 "lung_middle_lobe_right",
#                 "lung_lower_lobe_right"
#             ]
#
#             organ_mask = np.zeros_like(multilabel_mask)
#             for lung_lbl in lung_lbls:
#                 available_label_idx = available_labels[lung_lbl]
#                 organ_mask[multilabel_mask == available_label_idx] = 1
#
#             n = 1
#
#         else:
#             organ_mask = np.zeros_like(multilabel_mask)
#             available_label_idx = available_labels[lbl]
#             organ_mask[multilabel_mask == available_label_idx] = 1
#
#
#             if lbl == 'thyroid_gland':
#                 n = 2
#             else:
#                 n = 1
#
#         ### Remove small connected components
#         organ_mask = remove_small_cc(organ_mask, num_cc=n)
#
#         ### Fill holes
#         organ_mask = ndimage.binary_fill_holes(organ_mask)
#
#         ### Smooth boundaries
#         # radius = 10
#         # struct = ndimage.generate_binary_structure(rank=3, connectivity=1)  # 6-connectivity
#         # for _ in range(radius - 1):
#         #     struct = ndimage.binary_dilation(struct)
#         #
#         # # First remove small protrusions (opening), then fill small holes (closing)
#         # smooth = organ_mask.copy().astype(bool)
#         # for _ in range(10):
#         #     smooth = ndimage.binary_opening(smooth, structure=struct)
#         #     smooth = ndimage.binary_closing(smooth, structure=struct)
#
#         multilabel_mask_proc[organ_mask == 1] = organ_lbl_idx + 1
#
#     multilabel_mask_proc_nib = nib.Nifti1Image(multilabel_mask_proc, np.eye(4))
#     nib.save(multilabel_mask_proc_nib, os.path.join(OUTPUT_MASKS_DIR, mask_id + '.nii.gz'))
