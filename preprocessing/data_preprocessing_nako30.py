import os
import glob
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


IMAGE_DIR = '/home/kats/storage/staff/eytankats/data/nako_30/images/'
MASK_DIR = '/home/kats/storage/staff/eytankats/data/nako_30/masks_onehotencoded/'
OUTPUT_IMAGE_DIR = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_30/images_depth/'
OUTPUT_MASKS_BASE_DIR = '/home/kats/storage/staff/eytankats/projects/genseg/data/nako_30/'


labels_list = [
    "adrenal_gland_left",
    "adrenal_gland_right",
    "aorta",
    "autochthon_left",
    "autochthon_right",
    "clavicula_left",
    "clavicula_right",
    "esophagus",
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
    "prostate_uterus",
    "sacrum",
    "scapula_left",
    "scapula_right",
    "spleen",
    "stomach",
    "trachea",
    "urinary_bladder",
    "vertebrae_C1",
    "vertebrae_C2",
    "vertebrae_C3",
    "vertebrae_C4",
    "vertebrae_C5",
    "vertebrae_C6",
    "vertebrae_C7",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "vertebrae_T1",
    "vertebrae_T10",
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_T2",
    "vertebrae_T3",
    "vertebrae_T4",
    "vertebrae_T5",
    "vertebrae_T6",
    "vertebrae_T7",
    "vertebrae_T8",
    "vertebrae_T9",
]


output_labels_list = [
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


labels_dict = {}
for label_idx, label in enumerate(labels_list):
    labels_dict[label] = label_idx

if not os.path.exists(OUTPUT_IMAGE_DIR):
    os.makedirs(OUTPUT_IMAGE_DIR)

images_paths = sorted(glob.glob(IMAGE_DIR + '*.nii'))
masks_paths = sorted(glob.glob(MASK_DIR + '*.nii.gz'))

for image_path, mask_path in zip(images_paths, masks_paths):

    # load image and normalize it between 0 and 1
    img = nib.load(image_path).get_fdata()[..., 1]
    normalized_img = img / np.max(img)

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

    # save depth image
    np.save(os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(image_path))[:-4] + ".npy", depth_img)

    im = np.uint8(depth_img * 255)
    im = Image.fromarray(im)
    im.save(os.path.join(OUTPUT_IMAGE_DIR, os.path.basename(image_path))[:-4] + ".png")

    # load hot_encoded mask
    mask = nib.load(mask_path).get_fdata()

    # ensure that all labels are exist
    if mask.shape[-1] < 51:
        print(f'image {image_path} has {mask.shape[-1]} labels')
        continue

    for label_idx, label in enumerate(output_labels_list):

        depth_organ_mask = np.zeros((mask.shape[0], mask.shape[2]))
        if label == 'vertebrae':
            vertebrae_labels = [l for l in labels_list if label in l]

            for v_label in vertebrae_labels:

                # get organ mask
                organ_mask = mask[..., labels_dict[v_label]]

                # get depth image of organ and normalize by maximum depth of the depth body image
                depth_v_mask = np.argmax(organ_mask, axis=1)
                depth_v_mask = depth_v_mask / max_height
                depth_v_mask[depth_v_mask > 0] = 1 - depth_v_mask[depth_v_mask > 0]

                depth_organ_mask += depth_v_mask

            depth_organ_mask = np.clip(depth_organ_mask, 0, 1)

        else:
            # get organ mask
            organ_mask = mask[..., labels_dict[label]]

            # get depth image of organ and normalize by maximum depth of the depth body image
            depth_organ_mask = np.argmax(organ_mask, axis=1)
            depth_organ_mask = depth_organ_mask / max_height
            depth_organ_mask[depth_organ_mask > 0] = 1 - depth_organ_mask[depth_organ_mask > 0]

        # get output mask directory
        mask_output_dir = os.path.join(OUTPUT_MASKS_BASE_DIR, 'masks_' + label + '_depth')
        if not os.path.exists(mask_output_dir):
            os.makedirs(mask_output_dir)

        # save organ depth image
        im = np.uint8(depth_organ_mask * 255)
        im = Image.fromarray(im)
        im.save(os.path.join(mask_output_dir, os.path.basename(image_path))[:-4] + ".png")

        # get output visualisations directory
        vis_dir = os.path.join(mask_output_dir, 'visualizations')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # save visualization
        show_depth_img_with_organs(depth_img, depth_organ_mask, img_path=os.path.join(vis_dir, os.path.basename(image_path))[:-4] + ".png")











