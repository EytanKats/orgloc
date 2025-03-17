import os
import glob
import pickle
import numpy as np
import nibabel as nib

from PIL import Image
from scipy.ndimage import zoom
from scipy.ndimage import grey_opening, binary_opening


def show_depth_img_with_organs(depth_image, depth_mask, alpha=0.5, img_path=''):

    image_rgb = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
    image_rgb[:, :, 0] = depth_image
    image_rgb[:, :, 1] = depth_image
    image_rgb[:, :, 2] = depth_image
    image_rgb = np.uint8(image_rgb * 255)

    mask_rgb = np.zeros((depth_mask.shape[0], depth_mask.shape[1], 3))
    mask_rgb[:, :, 0] = depth_mask
    mask_rgb = np.uint8(mask_rgb * 255)

    blended_image = image_rgb * (1 - alpha) + mask_rgb * alpha
    blended_image = zoom(blended_image, (1, 2, 1), order=1)
    blended_image = np.rot90(blended_image)
    blended_image = np.uint8(blended_image)

    im = Image.fromarray(blended_image)
    im.save(img_path)


IMAGES_PATTERN = '/home/kats/storage/staff/eytankats/data/nako_10k/stitched/**/inp.nii.gz'
MASK_DIR = '/home/kats/storage/staff/eytankats/data/nako_10k/masks_totalmr/'
TEST_DATA_PICKLE_FILE = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_30/nako30_train_test_names.pkl'
OUTPUT_IMAGE_DIR = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_10k/images_depth/'
OUTPUT_MASKS_BASE_DIR = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_10k/'


labels_list = [
    "femur_left",
    "femur_right",
    "heart",
    "hip_left",
    "hip_right",
    "kidney_left",
    "kidney_right",
    "liver",
    "lung_left",
    "lung_right",
    "pancreas",
    "spleen",
    "stomach",
    "urinary_bladder",
    "vertebrae"
]

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)

images_paths = sorted(glob.glob(IMAGES_PATTERN))
with open(TEST_DATA_PICKLE_FILE, 'rb') as file:
    test_names = pickle.load(file)

for image_idx, image_path in enumerate(images_paths):

    # if file belongs to test set, skip it
    if image_path.split('/')[-2] + '_30' in test_names['test']['name_list']:
        continue

    sample_name = image_path.split('/')[-2]
    if not os.path.exists(os.path.join(MASK_DIR, sample_name, "femur_left" + '.nii.gz')):
        continue

    # load image and normalize it between 0 and 1
    img = nib.load(image_path).get_fdata()
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

    # get output image path
    image_output_path = os.path.join(OUTPUT_IMAGE_DIR, sample_name + '.png')

    # save depth image
    im = np.uint8(depth_img * 255)
    im = Image.fromarray(im)
    im.save(image_output_path)

    # prepare masks
    for label in labels_list:

        # get mask path
        mask_path = os.path.join(MASK_DIR, sample_name, label + '.nii.gz')

        # load binary mask
        mask = nib.load(mask_path).get_fdata()

        # flip image
        mask = np.flip(mask, axis=1)
        mask = np.flip(mask, axis=0)

        # get depth image of organ and normalize by maximum depth of the depth body image
        depth_organ_mask = np.argmax(mask, axis=1)
        depth_organ_mask = depth_organ_mask / max_height
        depth_organ_mask[depth_organ_mask > 0] = 1 - depth_organ_mask[depth_organ_mask > 0]

        # get output mask directory and path
        mask_output_dir = os.path.join(OUTPUT_MASKS_BASE_DIR, 'masks_' + label + '_depth')
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)

        mask_output_path = os.path.join(mask_output_dir, sample_name + '.png')

        # save organ depth image
        im = np.uint8(depth_organ_mask * 255)
        im = Image.fromarray(im)
        im.save(mask_output_path)

        if image_idx < 30:

            # get visualization image directory and path
            vis_output_dir = os.path.join(mask_output_dir, 'visualizations')
            if not os.path.exists(vis_output_dir):
                os.makedirs(vis_output_dir)

            vis_output_path = os.path.join(vis_output_dir, sample_name + '.png')

            # save visualization
            show_depth_img_with_organs(depth_img, depth_organ_mask, img_path=vis_output_path)











