import torch
import torch.nn.functional as F


def reconstruction_loss_mse(x, f):
    """
    Reconstruction loss
    :param x: ``Tensor([C, H, W])``, real image
    :param f: ``Tensor([C, H, W])``, decoded image
    :return: ``float``, divergence between decoded and real images
    """
    return F.mse_loss(x, f)


def hinge_loss(real, fake):
    """
    The hinge version loss
    :param real: ``Tensor([1, 5, 5])``
    :param fake: ``Tensor([1, 5, 5])``
    :return: ``float``, loss
    """
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


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
