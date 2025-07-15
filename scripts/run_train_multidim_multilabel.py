import sys
sys.path.append('./')
sys.path.append('../')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss
from monai.networks.utils import one_hot
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.swin_unetr import SwinUNETR

# Own Package
from models.segformer import Segformer
from models.unet_multidim import BasicUNet
from preprocessing.organ_labels_v2_volumetric import selected_organ_labels
from data.multidim_multilabel_dataset import Image_Dataset

from utils.get_logger import open_log
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.tools import seed_reproducer, save_checkpoint, get_cuda, print_options


def arg_parse() -> argparse.ArgumentParser.parse_args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/multidim_multilabel_unet_processedmasksv2.yaml', type=str, help='load the config file')
    args = parser.parse_args()
    return args


def run_trainer() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['output_path'] = os.path.join(configs['output_path'], time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '_' + configs['experiment_id'])
    configs['log_path'] = os.path.join(configs['output_path'], 'logs')

    wandb.init(
        project="organs_localization",
        name=os.path.basename(configs['experiment_id']),
        config=configs
    )

    # Output folder and save fig folder
    os.makedirs(configs['output_path'], exist_ok=True)
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
    train_dataset = Image_Dataset(configs['train_data_file_path'], images_dir=configs['images_dir'], masks_pattern=configs['masks_pattern'], labels_file=configs['labels_file'], stage='training', num_examples=configs['num_examples'], iterations=configs['iterations'] * configs['batch_size'])
    valid_dataset = Image_Dataset(configs['val_data_file_path'], images_dir=configs['images_dir'], masks_pattern=configs['masks_pattern'], labels_file=configs['labels_file'], stage='validation', num_examples=20)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True, drop_last=True, shuffle=True, )
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, num_workers=configs['num_workers'], pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    if configs['model'] == 'basic_unet':
        model = BasicUNet(in_channels=3, out_channels=len(selected_organ_labels) + 1, dropout=0.1)
    elif configs['model'] == 'attention_unet':
        model = AttentionUnet(spatial_dims=2, in_channels=3, out_channels=len(selected_organ_labels) + 1, channels=[32, 32, 64, 128, 256], strides=[2, 2, 2, 2, 2])
    elif configs['model'] == 'swin_unet':
        model = SwinUNETR(spatial_dims=2, img_size=256, in_channels=3, out_channels=len(selected_organ_labels) + 1, depths=[2, 2, 2, 2, 2], num_heads=[3, 6, 12, 24, 24])
    elif configs['model'] == 'segformer':
        model = Segformer(num_classes=len(selected_organ_labels) + 1)  # B0
        # model = Segformer(num_classes=len(ANATOMIES), dims=(64, 128, 320, 512), num_layers=(3, 3, 18, 3), decoder_dim=512)  # B3
    model = get_cuda(model)

    # Define optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs['lr'])
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1000, max_epochs=configs['iterations'])

    # Define loss functions
    weight = torch.tensor([
        1, # "background",
        4, # "spleen",
        6, # "kidney_right",
        6, # "kidney_left",
        1.5, # "liver",
        2, # "stomach",
        2, # "pancreas",
        1.5, # "lung_right",
        1.5, # "lung_left",
        5, # "trachea",
        12, # "thyroid_gland",
        4, # "duodenum",
        2, # "urinary_bladder",
        2.5, # "aorta",
        5, # "scapula_left",
        5, # "scapula_right",
        12, # "clavicula_left",
        12, # "clavicula_right",
        3.5, # "femur_left",
        3.5, # "femur_right",
        2.5, # "hip_left",
        2.5, # "hip_right",
        2, # "sacrum",
        10, # "vertebrae_L5",
        10, # vertebrae_L4",
        10, # "vertebrae_L3",
        10, # "vertebrae_L2",
        10, # "vertebrae_L1",
        10, # "vertebrae_T12",
        10, # "vertebrae_T11",
        10, # "vertebrae_T10",
        10, # "vertebrae_T9",
        10, # "vertebrae_T8",
        10, # "vertebrae_T7",
        10, # "vertebrae_T6",
        10, # "vertebrae_T5",
        10, # "vertebrae_T4",
        15, # "vertebrae_T3",
        10, # "vertebrae_T2",
        10, # "vertebrae_T1",
        1.5, # "heart"
    ], dtype=torch.float32)
    loss_DICE_CE = DiceCELoss(include_background=False, to_onehot_y=True, weight=weight.cuda())

    # For Tensorboard Visualization
    best_valid_loss = np.inf
    best_valid_dice = 0
    best_valid_dice_epoch = 0

    # Network training
    for epoch in range(1, 2):

        T_loss = []

        T_loss_valid = []

        T_Dice_valid = {}
        for anatomy in selected_organ_labels:
            T_Dice_valid[anatomy] = []

        ### Training phase
        iteration = 0
        for batch_data in tqdm(train_dataloader, desc='Train: '):
            img_rgb = batch_data['img'] / 255.0
            img_rgb = 2. * img_rgb - 1.

            seg_img = torch.stack(batch_data['seg'], dim=1).float()
            seg_img = seg_img.unsqueeze(1).float()

            model.train()

            # from matplotlib import pyplot as plt
            # plt.imshow((img_rgb[0].permute(1, 2, 0) + 1) / 2.)
            # plt.show()
            # plt.imshow(seg_img[0][0], cmap='gray')
            # plt.show()

            pred_seg = torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
            loss = loss_DICE_CE(pred_seg, get_cuda(seg_img))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            T_loss.append(loss.item())

            if iteration % 100 == 0:

                T_loss = np.mean(T_loss)

                logging.info("Train:")
                logging.info("loss_dice_ce: {:.4f}".format(T_loss))

                wandb.log({"train_loss_dice_ce": T_loss}, step=iteration)

                ### Validation phase
                for batch_data in tqdm(valid_dataloader, desc='Valid: '):
                    img_rgb = batch_data['img'] / 255.0
                    img_rgb = 2. * img_rgb - 1.

                    seg_img = torch.stack(batch_data['seg'], dim=1)
                    seg_img = seg_img.unsqueeze(1).float()

                    model.eval()

                    with torch.no_grad():

                        pred_seg = torch.nn.functional.sigmoid(model(get_cuda(img_rgb)))
                        loss= loss_DICE_CE(pred_seg, get_cuda(seg_img))

                        # calc dice
                        seg_img = one_hot(seg_img, len(selected_organ_labels) + 1)

                        pred_seg = torch.nn.functional.threshold(pred_seg,0.5, 0.).cpu()
                        reduce_axis = list(range(2, len(seg_img.shape)))

                        intersection = torch.sum(seg_img * pred_seg, dim=reduce_axis)
                        y_o = torch.sum(seg_img, dim=reduce_axis)
                        y_pred_o = torch.sum(pred_seg, dim=reduce_axis)
                        denominator = y_o + y_pred_o
                        dice_raw = (2.0 * intersection) / denominator
                        dice_value = dice_raw.mean(dim=0)

                        for anatomy_idx, anatomy in enumerate(selected_organ_labels):
                            T_Dice_valid[anatomy].append(dice_value[anatomy_idx + 1].item())

                        T_loss_valid.append(loss.item())

                T_loss_valid = np.mean(T_loss_valid)

                logging.info("Valid:")
                logging.info("loss_dice_ce: {:.4f}".format(T_loss_valid))

                dice_mean = 0
                for anatomy in selected_organ_labels:
                    dice_mean += np.mean(T_Dice_valid[anatomy])
                    logging.info(f'dice {anatomy}: {np.mean(T_Dice_valid[anatomy]):.4f}')
                    wandb.log({f'val_dice_{anatomy}': np.mean(T_Dice_valid[anatomy])}, step=iteration)
                dice_mean /= len(selected_organ_labels)
                wandb.log({"val_dice_mean": dice_mean}, step=iteration)

                wandb.log({"val_loss_dice_ce": T_loss_valid}, step=iteration)

                if dice_mean > best_valid_dice:
                    save_name = "best_valid_dice.pth"
                    save_checkpoint(model, save_name, configs['output_path'])
                    best_valid_dice = dice_mean
                    best_valid_dice_epoch = iteration
                    logging.info("Save best valid Dice !")

                if T_loss_valid < best_valid_loss:
                    save_name = "best_valid_loss.pth"
                    save_checkpoint(model, save_name, configs['output_path'])
                    best_valid_loss = T_loss_valid
                    logging.info("Save best valid Loss All !")

                # save_name = "{}_iteration_{:0>6}.pth".format('model', iteration)
                # save_checkpoint(model, save_name, configs['output_path'])

                logging.info('Current learning rate: {:.5f}'.format(scheduler.get_last_lr()[0]))
                logging.info('best valid dice: {:.4f} at iteration: {}'.format(best_valid_dice, best_valid_dice_epoch))
                logging.info('\n')

                T_loss = []

                T_loss_valid = []

                T_Dice_valid = {}
                for anatomy in selected_organ_labels:
                    T_Dice_valid[anatomy] = []

            iteration += 1

if __name__ == '__main__':
    run_trainer()
