import torch
import random
import numpy as np
from kornia.geometry.transform.crop.crop2d import center_crop


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
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

