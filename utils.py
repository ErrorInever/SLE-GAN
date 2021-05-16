import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from kornia.geometry.transform.crop.crop2d import center_crop
from mpl_toolkits.axes_grid1 import ImageGrid


def center_crop_img(tensor, size, mode):
    """
    Center crop of 2D tensor
    :param tensor: ``Tensor(B, C, H, W)``, 2D image tensor
    :param size: ``Tuple``, size crop
    :param mode: bilinear or nearest
    :return:
    """
    return center_crop(tensor, size, mode)


def get_random_noise():
    pass


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
