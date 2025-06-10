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
from tabulate import tabulate

from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader
from monai.networks.nets.basic_unet import BasicUnet

# Own Package
from data.multi_label_image_dataset import Image_Dataset
from preprocessing.organ_labels_v2 import selected_organ_labels

from utils.get_logger import open_log
from utils.tools import load_checkpoint, get_cuda, print_options, enable_dropout, mask_to_bbox_v2


def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/multilabel_basicunet_filteredaggmasksv2.yaml', type=str, help='load the config file')
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
    test_dataset = Image_Dataset(configs['test_data_file_path'], images_dir=configs['images_dir'], masks_dir=configs['masks_dir'], stage='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    model = BasicUnet(spatial_dims=2, in_channels=3, out_channels=len(selected_organ_labels), dropout=configs['dropout'])
    ckpt_path = os.path.join(configs['output_path'], configs['test_experiment_id'], 'checkpoints', configs['test_checkpoint'])
    model = load_checkpoint(model, ckpt_path)
    model = get_cuda(model)
    model.eval()

    name_list = []

    left = {}
    right = {}
    superior = {}
    inferior = {}
    for anatomy in selected_organ_labels:
        left[anatomy] = []
        right[anatomy] = []
        superior[anatomy] = []
        inferior[anatomy] = []
    left['mean'] = []
    right['mean'] = []
    superior['mean'] = []
    inferior['mean'] = []

    ### Validation phase
    for batch_idx, batch_data in tqdm(enumerate(test_dataloader), desc='Valid: '):
        img_rgb = batch_data['img'] / 255.0
        img_rgb = 2. * img_rgb - 1.

        seg_img = []
        for data in batch_data['seg']:
            seg_raw = data
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_raw[seg_raw > 0.] = 1.
            seg_img.append(torch.mean(seg_raw, dim=1, keepdim=False))
        seg_img = torch.stack(seg_img, dim=1).cuda()

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

            # calculate metrics
            left_mean = 0
            right_mean = 0
            superior_mean = 0
            inferior_mean = 0
            for anatomy_idx, anatomy in enumerate(selected_organ_labels):

                seg_anatomy = resize(seg_img[0, anatomy_idx, :, :].detach().cpu().numpy(), (480, 948), order=0, mode='constant')
                pred_anatomy = resize(pred_seg[0, anatomy_idx, :, :].detach().cpu().numpy(), (480, 948), order=1, mode='constant')

                bbox_seg = mask_to_bbox_v2(seg_anatomy)
                bbox_pred = mask_to_bbox_v2(np.uint8(pred_anatomy > 0.5))

                if bbox_seg is not None and bbox_pred is not None:
                    inferior_anatomy = bbox_seg['x1'] - bbox_pred['x1']
                    right_anatomy = bbox_seg['y1'] - bbox_pred['y1']
                    superior_anatomy = bbox_pred['x2'] - bbox_seg['x2']
                    left_anatomy = bbox_pred['y2'] - bbox_seg['y2']
                else:
                    left_anatomy = np.nan
                    right_anatomy = np.nan
                    superior_anatomy = np.nan
                    inferior_anatomy = np.nan

                left[anatomy].append(left_anatomy)
                left_mean += left_anatomy

                right[anatomy].append(right_anatomy)
                right_mean += right_anatomy

                superior[anatomy].append(superior_anatomy)
                superior_mean += superior_anatomy

                inferior[anatomy].append(inferior_anatomy)
                inferior_mean += inferior_anatomy

                # save images
                pred_anatomy_thr = np.float32(pred_anatomy > 0.5)
                if configs['save_imgs'] > 0 and configs['save_imgs'] > batch_idx:

                    img_to_plot = np.uint8(np.rot90(resize((img_rgb[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2, (480, 948), order=1, mode='constant')) * 255)
                    seg_to_plot = np.uint8(np.rot90(seg_anatomy) * 255)
                    pred_to_plot = np.uint8(np.rot90(pred_anatomy) * 255)
                    pred_to_plot_thr = np.uint8(np.rot90(pred_anatomy_thr) * 255)
                    overlay_to_plot = np.uint8(get_overlay(img_to_plot, seg_to_plot, pred_to_plot))
                    overlay_to_plot_thr = np.uint8(get_overlay(img_to_plot, seg_to_plot, pred_to_plot_thr))

                    seg_to_plot_rgb = np.zeros_like(img_to_plot)
                    seg_to_plot_rgb[:, :, 0] = seg_to_plot
                    seg_to_plot_rgb[:, :, 1] = seg_to_plot
                    seg_to_plot_rgb[:, :, 2] = seg_to_plot

                    pred_to_plot_rgb = np.zeros_like(img_to_plot)
                    pred_to_plot_rgb[:, :, 0] = pred_to_plot
                    pred_to_plot_rgb[:, :, 1] = pred_to_plot
                    pred_to_plot_rgb[:, :, 2] = pred_to_plot

                    pred_to_plot_thr_rgb = np.zeros_like(img_to_plot)
                    pred_to_plot_thr_rgb[:, :, 0] = pred_to_plot
                    pred_to_plot_thr_rgb[:, :, 1] = pred_to_plot
                    pred_to_plot_thr_rgb[:, :, 2] = pred_to_plot

                    stacked_img = np.concatenate((img_to_plot, seg_to_plot_rgb, pred_to_plot_rgb, pred_to_plot_thr_rgb, overlay_to_plot, overlay_to_plot_thr), axis=1)
                    img = Image.fromarray(stacked_img.astype(np.uint8))
                    img.save(os.path.join(configs['predictions_path'], name + '_' + anatomy + '.png'))

            left['mean'].append(left_mean / len(selected_organ_labels))
            right['mean'].append(right_mean / len(selected_organ_labels))
            superior['mean'].append(superior_mean / len(selected_organ_labels))
            inferior['mean'].append(inferior_mean / len(selected_organ_labels))

    # MEAN & Std Value
    name_list.extend(['Avg', 'Std'])
    left['mean'].extend([np.nanmean(left['mean']), np.nanstd(left['mean'], ddof=1)])
    right['mean'].extend([np.nanmean(right['mean']), np.nanstd(right['mean'], ddof=1)])
    superior['mean'].extend([np.nanmean(superior['mean']), np.nanstd(superior['mean'], ddof=1)])
    inferior['mean'].extend([np.nanmean(inferior['mean']), np.nanstd(inferior['mean'], ddof=1)])
    for anatomy_idx, anatomy in enumerate(selected_organ_labels):
        left[anatomy].extend([np.nanmean(left[anatomy]), np.nanstd(left[anatomy], ddof=1)])
        right[anatomy].extend([np.nanmean(right[anatomy]), np.nanstd(right[anatomy], ddof=1)])
        superior[anatomy].extend([np.nanmean(superior[anatomy]), np.nanstd(superior[anatomy], ddof=1)])
        inferior[anatomy].extend([np.nanmean(inferior[anatomy]), np.nanstd(inferior[anatomy], ddof=1)])

    # save csv
    csv_path = os.path.join(configs['output_path'], configs['test_experiment_id'], 'results.csv')

    df = pd.DataFrame({
        'name': name_list,
        'left_mean':  left['mean'],
        'right_mean': right['mean'],
        'superior_mean': superior['mean'],
        'inferior_mean': inferior['mean']
    })

    for anatomy in selected_organ_labels:
        df['left_' + anatomy] = left[anatomy]
        df['right_' + anatomy] = right[anatomy]
        df['superior_' + anatomy] = superior[anatomy]
        df['inferior_' + anatomy] = inferior[anatomy]

    df.to_csv(csv_path, index=False)

    print(tabulate(df.tail(2), headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    run_trainer()
