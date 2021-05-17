import torch
import random
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
from kornia.geometry.transform.crop.crop2d import center_crop
from mpl_toolkits.axes_grid1 import ImageGrid

logger = logging.getLogger(__name__)


def center_crop_img(tensor, size, mode):
    """
    Center crop image
    :param tensor: ``Tensor(B, C, H, W)``, 2D image tensor
    :param size: ``Tuple``, size crop
    :param mode: bilinear or nearest
    :return: ``Tensor(B, C, H, W)``, cropped image
    """
    return center_crop(tensor, size, mode)


def get_random_noise(size, dim, device):
    """
    Get random noise from normal distribution
    :param size: ``int``, number of samples (batch)
    :param dim: ``int``, dimension
    :param device: cuda or cpu device
    :return: Tensor([size, dim, 1, 1])
    """
    return torch.randn(size, dim, 1, 1).to(device)


def set_seed(seed):
    """
    For reproducible results
    :param seed: ``int``
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(gen, opt_gen, gen_scaler, dis, opt_dis, dis_scaler, fixed_noise, epoch):
    logger.info(f"=> Saving model")
    checkpoint = {
        'gen': gen.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'gen_scaler': gen_scaler.state_dict(),
        'dis': dis.state_dict(),
        'opt_dis': opt_dis.state_dict(),
        'dis_scaler': dis_scaler.state_dict(),
        'fixed_noise': fixed_noise,
        'epoch': epoch
    }
    save_checkpoint_path = os.path.join(cfg.OUT_DIR, f'{epoch}_ckpt.pth.tar')
    torch.save(checkpoint, save_checkpoint_path)
    logger.info(f"Success saved to {save_checkpoint_path}")


def load_checkpoint(ckpt, gen, opt_gen, gen_scaler, dis, opt_dis, dis_scaler, lr):
    logger.info(f"=> Loading checkpoint")
    checkpoint = torch.load(ckpt, map_location="cuda")
    gen.load_state_dict(checkpoint["gen"])
    opt_gen.load_state_dict(checkpoint["opt_gen"])
    gen_scaler.load_state_dict(checkpoint["gen_scaler"])
    dis.load_state_dict(checkpoint["dis"])
    opt_dis.load_state_dict(checkpoint["opt_dis"])
    dis_scaler.load_state_dict(checkpoint["dis_scaler"])
    fixed_noise = checkpoint['fixed_noise']
    epoch = checkpoint['epoch']

    for param_group in opt_gen.param_groups:
        param_group["lr"] = lr

    for param_group in opt_dis.param_groups:
        param_group["lr"] = lr

    logger.info(f"=>Success checkpoint loaded from {ckpt}")

    return fixed_noise, epoch


def show_batch(images, size=14, shape=(6, 6), save=None):
    """
    Show image grid from batch
    :param images: ``Tensor([N, C, H, W])`` images
    :param size: ``int``, size of image grid
    :param shape: ``Tuple or List``, shape of image grid, where first denotes number of rows,
    second denotes number of columns
    :param save: ``str``, path to save img
    """
    images_list = (images - images.min())/(images.max() - images.min())
    fig = plt.figure(figsize=(size, size))
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.04)
    for ax, image in zip(grid, images_list):
        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')

    if save:
        plt.savefig(save + 'name.png')

    plt.show()


def print_epoch_time(f):
    """Calculate time of each epoch and print it"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("epoch time: %2.1f min" % ((te-ts)/60))
        return result
    return timed
