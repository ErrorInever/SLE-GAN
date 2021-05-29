import torch
import torch.nn as nn
import numpy as np


def reconstruction_loss(x, f):
    """
    Reconstruction loss
    :param x: ``Tensor([C, H, W])``, real image
    :param f: ``Tensor([C, H, W])``, decoded image
    :return: ``float``, divergence between decoded and real images
    """
    l1 = nn.L1Loss(reduction='mean')
    return l1(x, f)


def hinge_adv_loss(real_fake_logits_real_images, real_fake_logits_fake_images):
    """
    the hinge version of the adversarial loss
    :param real_fake_logits_real_images: ``Tensor([1, 5, 5])``
    :param real_fake_logits_fake_images: ``Tensor([1, 5, 5])``
    :return: ``float``, discriminator loss
    """
    real_loss = -1 * torch.mean(np.minimum(0, -1 + real_fake_logits_real_images.detach().cpu()))
    fake_loss = -1 * torch.mean(np.minimum(0, -1 - real_fake_logits_fake_images.detach().cpu()))
    return real_loss + fake_loss
