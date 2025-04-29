import os
import cv2
import random
import logging
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

from preprocessing.organ_labels import selected_organ_labels


class Image_Dataset(Dataset):
    def __init__(self, data_file_path, images_dir, masks_dir, stage, num_examples=10000, iterations=None) -> None:
        super().__init__()

        self.mask_path = []
        self.img_path = os.path.join(images_dir, stage)
        for anatomy in selected_organ_labels:
            self.mask_path.append(os.path.join(masks_dir, stage, anatomy))

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

        index = random.randint(0, len(self.name_list) - 1)
        name = self.name_list[index]

        img_image = Image.open(os.path.join(self.img_path, name)).convert("RGB")
        img_data = np.array(img_image).astype(np.float32)

        seg_data = []
        for mask_path in self.mask_path:
            seg_image = Image.open(os.path.join(mask_path, name)).convert("RGB")
            seg_data.append(np.array(seg_image).astype(np.float32))

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

