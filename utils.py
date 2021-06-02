import torch
import random
import logging
import os
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
from kornia.geometry.transform.crop.crop2d import center_crop
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader, RandomSampler, Subset


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
    """
    Save checkpoint
    :param gen: ``Instance of models.model.Generator``, generator model
    :param opt_gen: ``Instance of torch.optim``, optimizer for generator
    :param gen_scaler: ``torch.cuda.amp.GradScaler()``, gradient scaler for generator
    :param dis: ``Instance of models.model.Discriminator``, discriminator model
    :param opt_dis: ``Instance of torch.optim``, optimizer for discriminator
    :param dis_scaler: ``torch.cuda.amp.GradScaler()``, gradient scaler for discriminator
    :param fixed_noise: ``Tensor([N, 1, Z, 1, 1])``, fixed noise for display result
    :param epoch: ``int``, current epoch
    """
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
    """
    Load checkpoint
    :param ckpt: ``str``, path to checkpoint.pth.tar
    :param gen: ``Instance of models.model.Generator``, generator model
    :param opt_gen: ``Instance of torch.optim``, optimizer for generator
    :param gen_scaler: ``torch.cuda.amp.GradScaler()``, gradient scaler for generator
    :param dis: ``Instance of models.model.Discriminator``, discriminator model
    :param opt_dis: ``Instance of torch.optim``, optimizer for discriminator
    :param dis_scaler: ``torch.cuda.amp.GradScaler()``, gradient scaler for discriminator
    :param lr: ``float``, learning rate
    :return: [``Tensor([N, 1, Z, 1, 1])``, ``int``], return fixed noise and epoch
    """
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


def get_sample_dataloader(dataset, num_samples, batch_size):
    """
    Get dataloader of random real data
    :param dataset: ``torch.data.dataset``, real dataset
    :param num_samples: ``int``, number images
    :param batch_size: ``int``, batch size
    :return: ``torch.data.dataloader``
    """
    sample_ds = Subset(dataset, np.arange(num_samples))
    sampler = RandomSampler(sample_ds)
    dataloader = DataLoader(sample_ds, sampler=sampler, batch_size=batch_size)
    return dataloader


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(critic, real, fake, device):
    """
    Calculate gradient penalty
    :param critic: Instance of discriminator (critic)
    :param real: ``Tensor([N, C, H, W])``, real images
    :param fake: ``Tensor([N, C, H, W])``, fake (generated) images
    :param device: torch device
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp
