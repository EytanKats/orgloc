import sys

sys.path.append('./')
sys.path.append('../')

# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import os
import pandas as pd
import nibabel as nib
from tabulate import tabulate

from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader
from monai.transforms import Resize

# Own Package
from models.unet_multidim import BasicUNet
from data.multidim_multilabel_dataset import Image_Dataset
from preprocessing.organ_labels_v2_volumetric import selected_organ_labels

from utils.get_logger import open_log
from utils.tools import load_checkpoint, get_cuda, print_options, enable_dropout, mask_to_bbox_volumetric


def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/multidim_multilabel_unet_filteredaggmasksv2.yaml', type=str, help='load the config file')
    args = parser.parse_args()
    return args


def get_overlay(image, mask_1, mask_2, alpha=0.5):
    image_rgb = np.copy(image)

    mask_rgb_1 = np.zeros((mask_1.shape[0], mask_1.shape[1], 3))
    mask_rgb_1[:, :, 0] = mask_1

    mask_rgb_2 = np.zeros((mask_2.shape[0], mask_2.shape[1], 3))
    mask_rgb_2[:, :, 1] = mask_2

    blended_image = image_rgb * (1 - alpha) + mask_rgb_1 * alpha + mask_rgb_2 * alpha

    return blended_image


def run_trainer() -> None:

    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['log_path'] = os.path.join(configs['output_path'], configs['test_experiment_id'], 'logs_test')
    configs['predictions_path'] = os.path.join(configs['output_path'], configs['test_experiment_id'], 'predictions')

    # Output folder and save fig folder
    os.makedirs(configs['predictions_path'], exist_ok=True)
    os.makedirs(configs['log_path'], exist_ok=True)

    # Set GPU ID
    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Get data loader
    test_dataset = Image_Dataset(configs['test_data_file_path'], images_dir=configs['images_dir'], masks_pattern=configs['masks_pattern'],
                                 labels_file=configs['labels_file'], stage='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    model = BasicUNet(in_channels=3, out_channels=len(selected_organ_labels) + 1, dropout=configs['dropout'])
    ckpt_path = os.path.join(configs['output_path'], configs['test_experiment_id'], 'checkpoints', configs['test_checkpoint'])
    model = load_checkpoint(model, ckpt_path)
    model = get_cuda(model)
    model.eval()

    name_list = []

    left = {}
    right = {}
    superior = {}
    inferior = {}
    anterior = {}
    posterior = {}
    for anatomy in selected_organ_labels:
        left[anatomy] = []
        right[anatomy] = []
        superior[anatomy] = []
        inferior[anatomy] = []
        anterior[anatomy] = []
        posterior[anatomy] = []
    left['mean'] = []
    right['mean'] = []
    superior['mean'] = []
    inferior['mean'] = []
    anterior['mean'] = []
    posterior['mean'] = []

    ### Validation phase
    for batch_idx, batch_data in tqdm(enumerate(test_dataloader), desc='Valid: '):
        img_rgb = batch_data['img'] / 255.0
        img_rgb = 2. * img_rgb - 1.

        seg_img = torch.stack(batch_data['seg'], dim=1).float()
        seg_img = seg_img.unsqueeze(1).float()

        name = batch_data['name'][0]
        name_list.append(name)

        # from matplotlib import pyplot as plt
        # plt.imshow((img_rgb[0].permute(1, 2, 0) + 1) / 2.)
        # plt.show()
        # plt.imshow(seg_img[0][7].cpu().numpy(), cmap='gray')
        # plt.show()

        with torch.no_grad():

            pred_seg = torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
            if configs['dropout'] > 0.:
                enable_dropout(model)
                for i in range(999):
                    pred_seg += torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
                pred_seg = pred_seg / 1000

            # Define resize transforms
            gt_resize_transform = Resize(spatial_size=(390, 480, 948), mode='nearest')
            pred_resize_transform = Resize(spatial_size=(390, 480, 948))

            # Save resized volume
            if configs['save_imgs'] > 0 and configs['save_imgs'] > batch_idx:
                volume_path = os.path.join(configs['volumes_dir'], name[:-4], 'wat.nii.gz')
                volume = nib.load(volume_path).get_fdata()

                # volume = np.flip(volume, axis=1)
                volume = np.flip(volume, axis=0)
                volume = torch.from_numpy(volume.copy()).float().permute(1, 0, 2)
                volume = pred_resize_transform(volume.unsqueeze(0)).squeeze().numpy()
                volume = nib.Nifti1Image(volume, np.eye(4))

                nib.save(volume, os.path.join(configs['predictions_path'], name[:-4] + '_vol.nii.gz'))

            # Calculate metrics
            left_mean = 0
            right_mean = 0
            superior_mean = 0
            inferior_mean = 0
            anterior_mean = 0
            posterior_mean = 0
            for anatomy_idx, anatomy in enumerate(selected_organ_labels):

                seg_anatomy = torch.zeros_like(seg_img)
                seg_anatomy[seg_img == anatomy_idx + 1] = 1
                seg_anatomy = seg_anatomy.squeeze()

                pred_anatomy = pred_seg[0, anatomy_idx + 1, :, :, :]

                seg_anatomy = gt_resize_transform(seg_anatomy.unsqueeze(0)).squeeze().numpy()
                pred_anatomy = pred_resize_transform(pred_anatomy.unsqueeze(0)).squeeze().detach().cpu().numpy()

                bbox_seg = mask_to_bbox_volumetric(seg_anatomy)
                bbox_pred = mask_to_bbox_volumetric(np.uint8(pred_anatomy > 0.5))

                if bbox_seg is not None and bbox_pred is not None:
                    inferior_anatomy = bbox_seg['x1'] - bbox_pred['x1']
                    right_anatomy = bbox_seg['y1'] - bbox_pred['y1']
                    superior_anatomy = bbox_pred['x2'] - bbox_seg['x2']
                    left_anatomy = bbox_pred['y2'] - bbox_seg['y2']
                    anterior_anatomy = bbox_seg['z1'] - bbox_pred['z1']
                    posterior_anatomy = bbox_pred['z2'] - bbox_seg['z2']
                else:
                    left_anatomy = np.nan
                    right_anatomy = np.nan
                    superior_anatomy = np.nan
                    inferior_anatomy = np.nan
                    anterior_anatomy = np.nan
                    posterior_anatomy = np.nan

                left[anatomy].append(left_anatomy)
                left_mean += left_anatomy

                right[anatomy].append(right_anatomy)
                right_mean += right_anatomy

                superior[anatomy].append(superior_anatomy)
                superior_mean += superior_anatomy

                inferior[anatomy].append(inferior_anatomy)
                inferior_mean += inferior_anatomy

                anterior[anatomy].append(anterior_anatomy)
                anterior_mean += anterior_anatomy

                posterior[anatomy].append(posterior_anatomy)
                posterior_mean += posterior_anatomy

                # save images
                pred_anatomy_thr = np.float32(pred_anatomy > 0.5)
                if configs['save_imgs'] > 0 and configs['save_imgs'] > batch_idx:

                    combined_anatomy = np.zeros_like(pred_anatomy)
                    combined_anatomy[(pred_anatomy_thr == 1) & (seg_anatomy == 1)] = 3
                    combined_anatomy[(seg_anatomy == 1) & (combined_anatomy != 3)] = 1
                    combined_anatomy[(pred_anatomy_thr == 1) & (combined_anatomy != 3)] = 2

                    combined_anatomy_nib = nib.Nifti1Image(combined_anatomy.astype(np.float32), np.eye(4))
                    nib.save(combined_anatomy_nib, os.path.join(configs['predictions_path'], name[:-4] + '_' + anatomy + '_cmb.nii.gz'))

                    pred_anatomy_nib = nib.Nifti1Image(pred_anatomy_thr.astype(np.float32), np.eye(4))
                    nib.save(pred_anatomy_nib, os.path.join(configs['predictions_path'], name[:-4] + '_' + anatomy + '_pred.nii.gz'))

                    seg_anatomy_nib = nib.Nifti1Image(seg_anatomy, np.eye(4))
                    nib.save(seg_anatomy_nib, os.path.join(configs['predictions_path'], name[:-4] + '_' + anatomy + '_gt.nii.gz'))

                    img_to_plot = np.uint8(np.rot90(resize((img_rgb[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2, (480, 948), order=1, mode='constant')) * 255)
                    img.save(os.path.join(configs['predictions_path'], name + '.png'))

            left['mean'].append(left_mean / len(selected_organ_labels))
            right['mean'].append(right_mean / len(selected_organ_labels))
            superior['mean'].append(superior_mean / len(selected_organ_labels))
            inferior['mean'].append(inferior_mean / len(selected_organ_labels))
            anterior['mean'].append(anterior_mean / len(selected_organ_labels))
            posterior['mean'].append(posterior_mean / len(selected_organ_labels))

    # MEAN & Std Value
    name_list.extend(['Avg', 'Std'])
    left['mean'].extend([np.nanmean(left['mean']), np.nanstd(left['mean'], ddof=1)])
    right['mean'].extend([np.nanmean(right['mean']), np.nanstd(right['mean'], ddof=1)])
    superior['mean'].extend([np.nanmean(superior['mean']), np.nanstd(superior['mean'], ddof=1)])
    inferior['mean'].extend([np.nanmean(inferior['mean']), np.nanstd(inferior['mean'], ddof=1)])
    anterior['mean'].extend([np.nanmean(anterior['mean']), np.nanstd(anterior['mean'], ddof=1)])
    posterior['mean'].extend([np.nanmean(posterior['mean']), np.nanstd(posterior['mean'], ddof=1)])

    for anatomy_idx, anatomy in enumerate(selected_organ_labels):
        left[anatomy].extend([np.nanmean(left[anatomy]), np.nanstd(left[anatomy], ddof=1)])
        right[anatomy].extend([np.nanmean(right[anatomy]), np.nanstd(right[anatomy], ddof=1)])
        superior[anatomy].extend([np.nanmean(superior[anatomy]), np.nanstd(superior[anatomy], ddof=1)])
        inferior[anatomy].extend([np.nanmean(inferior[anatomy]), np.nanstd(inferior[anatomy], ddof=1)])
        anterior[anatomy].extend([np.nanmean(anterior[anatomy]), np.nanstd(anterior[anatomy], ddof=1)])
        posterior[anatomy].extend([np.nanmean(posterior[anatomy]), np.nanstd(posterior[anatomy], ddof=1)])

    # save csv
    csv_path = os.path.join(configs['output_path'], configs['test_experiment_id'], 'results.csv')

    df = pd.DataFrame({
        'name': name_list,
        'left_mean':  left['mean'],
        'right_mean': right['mean'],
        'superior_mean': superior['mean'],
        'inferior_mean': inferior['mean'],
        'anterior_mean': anterior['mean'],
        'posterior_mean': posterior['mean']
    })

    for anatomy in selected_organ_labels:
        df['left_' + anatomy] = left[anatomy]
        df['right_' + anatomy] = right[anatomy]
        df['superior_' + anatomy] = superior[anatomy]
        df['inferior_' + anatomy] = inferior[anatomy]
        df['anterior_' + anatomy] = anterior[anatomy]
        df['posterior_' + anatomy] = posterior[anatomy]

    df.to_csv(csv_path, index=False)

    print(tabulate(df.tail(2), headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    run_trainer()
