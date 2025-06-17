import os
import cv2
import glob
import json
import random
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import albumentations as A

from PIL import Image
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from preprocessing.organ_labels_v2_volumetric import selected_organ_labels


class Image_Dataset(Dataset):
    def __init__(self, data_file_path, images_dir, masks_pattern, labels_file, stage, num_examples=10000, iterations=None) -> None:
        super().__init__()

        self.mask_path = []
        self.img_path = os.path.join(images_dir, stage)

        masks_paths = sorted(glob.glob(masks_pattern))
        masks_ids = [os.path.basename(mask_path)[:6] for mask_path in masks_paths]
        self.masks_map = dict(zip(masks_ids, masks_paths))

        self.available_labels = json.load(open(labels_file))["labels"]

        self.iterations = iterations
        self.img_size = 256
        self.stage = stage
        self.name_list = pd.read_csv(data_file_path)['id'].tolist()[:num_examples]
        self.transform = self.get_transforms()
        logging.info('{} set num: {}'.format(stage, len(self.name_list)))

    def get_transforms(self):
        if self.stage == 'train':
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                # A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                A.ToFloat(max_value=255.0),
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])
        return transforms

    def __getitem__(self, index):

        if self.stage == 'training':
            index = random.randint(0, len(self.name_list) - 1)

        name = self.name_list[index]

        img_image = Image.open(os.path.join(self.img_path, name)).convert("RGB")
        img_data = np.array(img_image).astype(np.float32)

        # load multilabel mask
        multilabel_mask = nib.load(self.masks_map[name[:-4]]).get_fdata()

        # flip mask
        multilabel_mask = np.flip(multilabel_mask, axis=1)
        # multilabel_mask = np.flip(multilabel_mask, axis=0)
        multilabel_mask = zoom(multilabel_mask, zoom=(1, 64 / multilabel_mask.shape[1], 1), order=0)

        selected_labels_mask = np.zeros_like(multilabel_mask)
        for anatomy_idx, anatomy in enumerate(selected_organ_labels):
            organ_label_idx = self.available_labels[anatomy]
            selected_labels_mask[multilabel_mask == organ_label_idx] = anatomy_idx + 1

        seg_data = [selected_labels_mask[:, slice_idx, :] for slice_idx in range(selected_labels_mask.shape[1]) ]
        seg_data = [zoom(seg_slice, zoom=(300 / seg_slice.shape[0], 631 / seg_slice.shape[1]), order=0) for seg_slice in seg_data]
        # seg_data = [np.flip(seg_slice, axis=0) for seg_slice in seg_data]

        augmented = self.transform(image=img_data, masks=seg_data)

        aug_img = augmented['image']
        aug_seg = augmented['masks']

        return {
            'name': name,
            'img': aug_img,
            'seg': aug_seg
        }

    def __len__(self):
        if self.iterations is None:
            return len(self.name_list)
        else:
            return self.iterations

