import os
import cv2
import random
import pickle
import logging
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


ANATOMIES = [
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


class Image_Dataset(Dataset):
    def __init__(self, pickle_file_path, stage='train', num_examples=10000, iterations=None) -> None:
        super().__init__()
        with open(pickle_file_path, 'rb') as file:
            loaded_dict = pickle.load(file)

        self.mask_path = []
        self.mask_smooth_path = []
        self.img_path = os.path.join(os.path.dirname(pickle_file_path), 'images_depth')
        for anatomy in ANATOMIES:
            self.mask_path.append(os.path.join(os.path.dirname(pickle_file_path), 'masks_' + anatomy + '_depth'))

        self.iterations = iterations
        self.img_size = 256
        self.stage = stage
        self.name_list = loaded_dict[stage]['name_list'][:num_examples]
        self.transform = self.get_transforms()
        logging.info('{} set num: {}'.format(stage, len(self.name_list)))

        del loaded_dict

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

        img_image = Image.open(os.path.join(self.img_path, name + '.png')).convert("RGB")
        img_data = np.array(img_image).astype(np.float32)

        seg_data = []
        for mask_path in self.mask_path:
            seg_image = Image.open(os.path.join(mask_path, name + '.png')).convert("RGB")
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

