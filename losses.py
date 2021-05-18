import torch
import torch.nn as nn


def reconstruction_loss(f, x):
    """
    Reconstruction loss
    :param f: ``Tensor([C, H, W])``, decoded image
    :param x: ``Tensor([C, H, W])``, real image
    :return: ``float``, divergence between decoded and real images
    """
    return nn.L1Loss(f, x)


def hinge_adv_loss(real_fake_logits_real, real_fake_logits_fake):
    """
    the hinge version of the adversarial loss
    :param real_fake_logits_real: ``Tensor([1, 5, 5])``
    :param real_fake_logits_fake: ``Tensor([1, 5, 5])``
    :return: ``float``, discriminator loss
    """
    real_loss = -1 * torch.mean(torch.minimum(torch.Tensor([0.0]), -1 + real_fake_logits_real))
    fake_loss = -1 * torch.mean(torch.minimum(torch.Tensor([0.0]), -1 - real_fake_logits_fake))
    return real_loss + fake_loss

