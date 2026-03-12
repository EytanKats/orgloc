import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import yaml

# Add project root to path
sys.path.append('./')

from data.multidim_multilabel_dataset import Image_Dataset

def main():
    # Load config to get paths
    config_path = '../configs/multidim_multilabel_unet_processedmasksv2.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    # Instantiate dataset in training mode to see augmentations
    # Use small num_examples and iterations for validation
    try:
        dataset = Image_Dataset(
            data_file_path=configs['train_data_file_path'],
            images_dir=configs['images_dir'],
            masks_pattern=configs['masks_pattern'],
            labels_file=configs['labels_file'],
            stage='training', # This triggers get_transforms() to use train augmentations
            num_examples=10
        )
    except Exception as e:
        print(f"Error creating dataset: {e}. Check if paths in config are accessible on this machine.")
    else:
        # Force stage to 'training' to also trigger cloth folding AND random index in __getitem__
        dataset.stage = 'training' 

        # Get a sample
        # Note: stage='training' makes __getitem__ pick a random index
        sample = dataset[0]
        name = sample['name']
        img = sample['img'] # Tensor (C, H, W)
        seg = sample['seg'] # List of Tensors (H, W)

        # Convert image for plotting
        # img is (3, 256, 256) after ToTensorV2
        img_np = img.permute(1, 2, 0).numpy()
        
        # Reconstruct 3D mask for projection
        seg_3d = np.stack([s.numpy() if torch.is_tensor(s) else s for s in seg], axis=0) # (64, 256, 256)
    
        # Max projection across depth (axis 0)
        seg_proj = np.max(seg_3d, axis=0)
    
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Normalize image to [0, 1] for plotting if it isn't already
        if img_np.max() > 1.1:
            img_np = img_np / 255.0
    
        axes[0].imshow(img_np)
        axes[0].set_title(f"Augmented Image: {name}")
        axes[0].axis('off')
    
        # Use tab20 for labels
        axes[1].imshow(seg_proj, cmap='tab20', interpolation='nearest')
        axes[1].set_title("Augmented Mask (Max Projection)")
        axes[1].axis('off')
    
        plt.tight_layout()
        plt.show()
        print("Validation image saved to augmentation_validation.png")

        # Check for consistency: print shapes
        print(f"Image shape: {img_np.shape}")
        print(f"3D Mask shape: {seg_3d.shape}")
        print(f"Max label in mask: {np.max(seg_3d)}")

if __name__ == "__main__":
    main()
