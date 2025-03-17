# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import os
import pandas as pd
from tabulate import tabulate

from skimage.transform import resize
from scipy.ndimage import zoom
import sklearn.metrics as metrics
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from monai.networks.nets.basic_unet import BasicUnet

# Own Package
from data.multi_label_image_dataset import Image_Dataset, ANATOMIES
from utils.metrics import compute_surface_distances, compute_average_surface_distance
from utils.tools import seed_reproducer, load_checkpoint, get_cuda, print_options, enable_dropout, mask_to_bbox, get_iou
from utils.get_logger import open_log

def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nako1000_valid.yaml',
                        type=str, help='load the config file')
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
    configs['log_path'] = os.path.join(configs['snapshot_path'], 'logs')
    
    # Output folder and save fig folder
    os.makedirs(configs['snapshot_path'], exist_ok=True)
    os.makedirs(configs['save_seg_img_path'], exist_ok=True)
    os.makedirs(configs['log_path'], exist_ok=True)

    # Set GPU ID
    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    # Fix seed (for repeatability)
    seed_reproducer(configs['seed'])

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Get data loader
    valid_dataset = Image_Dataset(configs['pickle_file_path'], stage='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    model = BasicUnet(spatial_dims=2, in_channels=3, out_channels=len(ANATOMIES), dropout=configs['dropout'])
    model = load_checkpoint(model, configs['model_weight'])
    model = get_cuda(model)
    model.eval()
    enable_dropout(model)

    name_list = []

    dice05 = {}
    sd05 = {}
    auc = {}
    for anatomy in ANATOMIES:
        dice05[anatomy] = []
        sd05[anatomy] = []
        auc[anatomy] = []
    dice05['mean'] = []
    sd05['mean'] = []
    auc['mean'] = []

    ### Validation phase
    for batch_data in tqdm(valid_dataloader, desc='Valid: '):
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
        # plt.imshow(seg_img[0][0], cmap='gray')
        # plt.show()

        with torch.no_grad():

            pred_seg = torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
            if configs['dropout'] > 0.:
                for i in range(999):
                    pred_seg += torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
                pred_seg = pred_seg / 1000

            # calculate metrics
            dice_mean = 0
            sd_mean = 0
            auc_mean = 0

            for anatomy_idx, anatomy in enumerate(ANATOMIES):

                seg_anatomy = resize(seg_img[0, anatomy_idx, :, :].detach().cpu().numpy(), (480, 948), order=0, mode='constant')
                pred_anatomy = resize(pred_seg[0, anatomy_idx, :, :].detach().cpu().numpy(), (480, 948), order=1, mode='constant')

                # precision, recall, threshold = metrics.precision_recall_curve(seg_anatomy.flatten(), pred_anatomy.flatten())
                # roc_auc = metrics.auc(recall, precision) * 100

                bbox_seg = mask_to_bbox(seg_anatomy)
                bbox_pred = mask_to_bbox(np.uint8(pred_anatomy > 0.5))

                if bbox_seg is not None and bbox_pred is not None:
                    roc_auc = np.max([
                        bbox_pred['x1'] - bbox_seg['x1'],
                        bbox_pred['y1'] - bbox_pred['y1'],
                        bbox_seg['x2'] - bbox_pred['x2'],
                        bbox_seg['y2'] - bbox_pred['y2'],
                        0
                    ])
                else:
                    roc_auc = np.nan

                pred_anatomy_thr = np.float32(pred_anatomy > 0.5)

                # calculate surface distance
                dist = compute_surface_distances(seg_anatomy > 0.5, pred_anatomy > 0.5, (1, 1))
                avg_dist = compute_average_surface_distance(dist)[0]

                # calculate dice
                intersection = np.sum(seg_anatomy * pred_anatomy_thr)
                y_o = np.sum(seg_anatomy)
                y_pred_o = np.sum(pred_anatomy_thr)
                denominator = y_o + y_pred_o
                dice_value = (2.0 * intersection) / denominator

                auc[anatomy].append(roc_auc)
                auc_mean += roc_auc

                dice05[anatomy].append(dice_value.item())
                dice_mean += dice_value.item()

                sd05[anatomy].append(avg_dist)
                sd_mean += avg_dist

                # save images
                if configs['save_imgs']:
                    img_to_plot = np.uint8(np.rot90(zoom((img_rgb[0].permute(1, 2, 0).cpu().detach().numpy() + 1) / 2, zoom=(1, 2, 1), order=1)) * 255)
                    seg_to_plot = np.uint8(np.rot90(zoom(seg_anatomy.cpu().detach().numpy(), zoom=(1, 2), order=0)) * 255)
                    pred_to_plot = np.uint8(np.rot90(zoom(pred_anatomy.cpu().detach().numpy(), zoom=(1, 2), order=0)) * 255)
                    pred_to_plot_thr = np.uint8(np.rot90(zoom(pred_anatomy_thr.cpu().detach().numpy() > 0.5, zoom=(1, 2), order=0)) * 255)
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
                    img.save(os.path.join(configs['save_seg_img_path'], name + '_' + anatomy + '.png'))

            dice05['mean'].append(dice_mean / len(ANATOMIES))
            sd05['mean'].append(sd_mean / len(ANATOMIES))
            auc['mean'].append(auc_mean / len(ANATOMIES))

    # MEAN & Std Value
    name_list.extend(['Avg', 'Std'])
    dice05['mean'].extend([np.nanmean(dice05['mean']), np.nanstd(dice05['mean'], ddof=1)])
    sd05['mean'].extend([np.nanmean(sd05['mean']), np.nanstd(sd05['mean'], ddof=1)])
    auc['mean'].extend([np.nanmean(auc['mean']), np.nanstd(auc['mean'], ddof=1)])
    for anatomy_idx, anatomy in enumerate(ANATOMIES):
        dice05[anatomy].extend([np.nanmean(dice05[anatomy]), np.nanstd(dice05[anatomy], ddof=1)])
        sd05[anatomy].extend([np.nanmean(sd05[anatomy]), np.nanstd(sd05[anatomy], ddof=1)])
        auc[anatomy].extend([np.nanmean(auc[anatomy]), np.nanstd(auc[anatomy], ddof=1)])

    # save csv
    csv_path = os.path.join(configs['snapshot_path'], 'results.csv')

    df = pd.DataFrame({
        'name': name_list,
        'dice_mean':  dice05['mean'],
        'sd_mean': sd05['mean'],
        'auc_mean': auc['mean']
    })

    for anatomy in ANATOMIES:
        df['dice_' + anatomy] = dice05[anatomy]
        df['sd_' + anatomy] = sd05[anatomy]
        df['auc_' + anatomy] = auc[anatomy]

    df.to_csv(csv_path, index=False)

    print(tabulate(df.tail(2), headers='keys', tablefmt='psql'))

if __name__ == '__main__':
    run_trainer()
