import os
import glob
import pandas as pd


IMAGE_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/images_depth'
MASK_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_volumetric_preprocessed_v2'
PARTITION = 'test'

# find data samples for which both image and mask are successfully extracted
image_paths = glob.glob(os.path.join(IMAGE_DIR, PARTITION, '*.png'))
masks_paths = glob.glob(os.path.join(MASK_DIR, '*.nii.gz'))

image_ids = [os.path.basename(n)[:6] for n in image_paths]
mask_ids = [os.path.basename(n)[:6] for n in masks_paths]

ids = list(set(image_ids) & set(mask_ids))

# create data frame to keep information about masks
df = pd.DataFrame({'id':ids})

df.to_csv(os.path.join(MASK_DIR, f'{PARTITION}_masks_list.csv'), index=False)
print(f'Number of data samples: {len(df)}')








