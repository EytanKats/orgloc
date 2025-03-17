import os
import json
import glob

import numpy as np
import nibabel as nib

from tqdm import tqdm
from PIL import Image
from nibabel import processing
from scipy.ndimage import zoom
from scipy.ndimage import grey_opening, binary_opening

DATA_SPLIT_FILE = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_10k/nako_dataset_split.json'
IMAGES_PATTERN_1K = '/home/kats/storage/staff/eytankats/data/nako_1000/nii_allmod_stitched/**/inp.nii.gz'
IMAGES_PATTERN_10K = '/home/kats/storage/staff/eytankats/data/nako_10k/stitched/**/inp.nii.gz'
OUTPUT_IMAGE_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/depth_images_test/'

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)

images_paths = sorted(glob.glob(IMAGES_PATTERN_10K) + glob.glob(IMAGES_PATTERN_1K))
images_ids = [image_path.split('/')[-2] for image_path in images_paths]

test_images_ids = [test_image_name[:6] for test_image_name in json.load(open(DATA_SPLIT_FILE))['test']]

for test_image_id in tqdm(test_images_ids):

    image_path = images_paths[images_ids.index(test_image_id)]

    # load image, resample and normalize it between 0 and 1
    img = nib.load(image_path).get_fdata()
    normalized_img = img / np.max(img)

    # rescale image to 1.5x1.5x1.5
    normalized_img = zoom(normalized_img, zoom=(300 / 320, 244 / 260, 631 / 316), order=1)

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

    # get output image path
    image_output_path = os.path.join(OUTPUT_IMAGE_DIR, test_image_id + '.png')

    # save depth image
    im = np.uint8(depth_img * 255)
    im = Image.fromarray(im)
    im.save(image_output_path)