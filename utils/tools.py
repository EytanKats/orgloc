import torch
import os
import random
import numpy as np
import logging

from skimage.measure import label, regionprops


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if os.path.isfile(path):
        logging.info("=> loading checkpoint '{}'".format(path))
        
        # remap everthing onto CPU 
        state = torch.load(path, map_location=lambda storage, location: storage)

        # load weights
        model.load_state_dict(state['model'])
        logging.info("Loaded")
    else:
        model = None
        logging.info("=> no checkpoint found at '{}'".format(path))
    return model


def save_checkpoint(model: torch.nn.Module, save_name: str, path: str) -> None:
    model_savepath = os.path.join(path, 'checkpoints')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    file_name = os.path.join(model_savepath, save_name)
    torch.save({
        'model': model.state_dict(),
    },file_name)
    logging.info("save model to {}".format(file_name))


def adjust_learning_rate(optimizer, initial_lr, epoch, reduce_epoch, decay=0.5):
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('Change Learning Rate to {}'.format(lr))
    return lr


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bbox = {}

    lbl = label(mask)
    if np.max(lbl) == 0:
        return None

    largestCC = np.uint8(lbl == np.argmax(np.bincount(lbl.flat)[1:]) + 1)

    props = regionprops(largestCC)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bbox['x1'] = x1
        bbox['y1'] = y1
        bbox['x2'] = x2
        bbox['y2'] = y2

    return bbox

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def disable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

def print_options(configs):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in configs.items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logging.info(message)

    # save to the disk
    file_name = os.path.join(configs['log_path'], '{}_configs.txt'.format(configs['phase']))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')