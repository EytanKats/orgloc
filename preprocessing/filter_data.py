import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from preprocessing.organ_labels import selected_organ_labels


IMAGE_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/images_depth'
MASK_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_projection'
PARTITION = 'training'

# find data samples for which both image and mask are successfully extracted
image_paths = glob.glob(os.path.join(IMAGE_DIR, PARTITION, '*.png'))
masks_paths = glob.glob(os.path.join(MASK_DIR, PARTITION, selected_organ_labels[0], '*.png'))

image_ids = [os.path.basename(n) for n in image_paths]
mask_ids = [os.path.basename(n) for n in masks_paths]

ids = list(set(image_ids) & set(mask_ids))

# create data frame to keep onformation about masks
df = pd.DataFrame({'id':ids})

# iterate over masks of selected organs and calculate organ area and number of connected components
for label in selected_organ_labels:

    # create lists to collect the information
    area_list = []
    cnt_list = []

    # iterate over data samples
    for data_id in tqdm(ids):

        # Load mask and binarize it
        mask = Image.open(os.path.join(MASK_DIR, PARTITION, label, os.path.basename(data_id)))
        mask = np.array(mask)
        mask[mask > 0] = 1

        # get connected components and calculate the area of the biggest one
        labeled_array, num_features = ndimage.label(mask)
        area = np.sum(labeled_array == 1)

        # append values to corresponding lists
        area_list.append(area)
        cnt_list.append(num_features)

    # create columns in the dataframe
    df[f'area_{label}'] = area_list
    df[f'cnt_{label}'] = cnt_list

# save original dataframe
df.to_csv(os.path.join(MASK_DIR, f'{PARTITION}_masks_info.csv'), index=False)
print(f'Original number of data samples: {len(df)}')

# filter data based on number of connected components
for label in selected_organ_labels:
    df = df[df[f'cnt_{label}'] == 1]
print(f'cc-filtered number of data samples: {len(df)}')

# filter data based on area of the segmented region
for label in selected_organ_labels:
    bottom_5_percent_threshold = df[f'area_{label}'].quantile(0.01)  # calculate the 5th percentile
    df = df[df[f'area_{label}'] > bottom_5_percent_threshold]
print(f'area-filtered number of data samples: {len(df)}')
df.to_csv(os.path.join(MASK_DIR, f'{PARTITION}_masks_filtered_info.csv'), index=False)








