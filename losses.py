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
    the hinge version loss
    :param real: ``Tensor([1, 5, 5])``
    :param fake: ``Tensor([1, 5, 5])``
    :return: ``float``, loss
    """
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()
