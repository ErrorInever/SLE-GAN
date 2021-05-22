import torch
import torch.nn as nn


def reconstruction_loss(x, f):
    """
    Reconstruction loss
    :param f: ``Tensor([C, H, W])``, decoded image
    :param x: ``Tensor([C, H, W])``, real image
    :return: ``float``, divergence between decoded and real images
    """
    l1 = nn.L1Loss()
    return l1(x, f)


def hinge_adv_loss(real_fake_logits_real, real_fake_logits_fake):
    """
    the hinge version of the adversarial loss
    :param real_fake_logits_real: ``Tensor([1, 5, 5])``
    :param real_fake_logits_fake: ``Tensor([1, 5, 5])``
    :return: ``float``, discriminator loss
    """
    real_loss = -1 * torch.mean(torch.minimum(torch.Tensor([0.0]), -1 + real_fake_logits_real.detach().cpu()))
    fake_loss = -1 * torch.mean(torch.minimum(torch.Tensor([0.0]), -1 - real_fake_logits_fake.detach().cpu()))
    return real_loss + fake_loss
