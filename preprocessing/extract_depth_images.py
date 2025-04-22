import os
import json
import glob

import numpy as np
import nibabel as nib

from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
from scipy.ndimage import grey_opening, binary_opening

DATA_SPLIT_FILE = '/home/kats/storage/staff/eytankats/data/nako_10k/nako_dataset_split.json'
IMAGES_PATTERN = '/home/kats/storage/staff/eytankats/data/nako_10k/images_mri_stitched/**/inp.nii.gz'
OUTPUT_IMAGE_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/images_depth'
DATA_PARTITION = 'training'

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)

images_paths = sorted(glob.glob(IMAGES_PATTERN))
images_ids = [image_path.split('/')[-2] for image_path in images_paths]
partition_images_ids = [image_name[:6] for image_name in json.load(open(DATA_SPLIT_FILE))[DATA_PARTITION]]

for image_id in tqdm(partition_images_ids):

    image_path = images_paths[images_ids.index(image_id)]

    # load image and normalize it between 0 and 1
    try:
        img = nib.load(image_path).get_fdata()
    except:
        print(f'Cannot load {image_path}, skipping.')
        continue

    normalized_img = img / np.max(img)

    # flip image
    normalized_img = np.flip(normalized_img, axis=1)
    normalized_img = np.flip(normalized_img, axis=0)

    # get maximum depth
    max_height = img.shape[1]

    # get nonzero mask by threshold
    non_zero_mask = normalized_img > 0.02

    # clean nonzero mask by morphological opening
    non_zero_mask = binary_opening(non_zero_mask, iterations=5)

    # get normalized depth_image
    depth_img = np.argmax(non_zero_mask, axis=1)
    depth_img = depth_img / max_height
    depth_img[depth_img > 0] = 1 - depth_img[depth_img > 0]

    # clean depth image by thresholding
    depth_img[depth_img < 0.3] = 0

    # clean depth image by morphological opening
    depth_img = grey_opening(depth_img, size=(11, 11))

    # rescale image to 1.5x1.5
    depth_img = zoom(depth_img, zoom=(300 / 320, 631 / 316), order=1)

    # get output image path
    image_output_path = os.path.join(OUTPUT_IMAGE_DIR, DATA_PARTITION,  image_id + '.png')

    # save depth image
    im = np.uint8(depth_img * 255)
    im = Image.fromarray(im)
    im.save(image_output_path)