import logging
import os
import errno
import wandb
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from config import cfg


logger = logging.getLogger(__name__)


class MetricLogger:
    """Metric class"""
    def __init__(self, project_version_name):
        """
        :param project_version_name: ``str``, name of current version of project
        """
        self.project_version_name = project_version_name
        self.data_subdir = f"{os.path.join(cfg.OUT_DIR, self.project_version_name)}/fixed_images"

        if cfg.RESUME_ID:
            wandb_id = cfg.RESUME_ID
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project='SLE-GC-GAN', name=project_version_name, resume=True)
        wandb.config.update({
            'img_size': cfg.IMG_SIZE,
            'generator_type': cfg.GEN_TYPE,
            'learning_rate': cfg.LEARNING_RATE,
            'z_dimension': cfg.Z_DIMENSION,
            'model_depth': cfg.IN_CHANNELS,
            'batch_sizes': cfg.BATCH_SIZE,
        })

    def log(self, g_loss, d_loss, logits_loss, i_recons_loss, i_part_recons_loss):
        """
        Logging all losses
        :param g_loss: ``torch.Variable``, generator loss
        :param d_loss: ``torch.Variable``, discriminator loss
        :param logits_loss: ``torch.Variable``, discriminator hinge loss
        :param i_recons_loss: ``torch.Variable``, reconstruction on real images loss
        :param i_part_recons_loss: ``torch.Variable``, reconstruction on part of real images loss
        """
        if isinstance(g_loss, torch.autograd.Variable):
            g_loss = g_loss.item()
        if isinstance(d_loss, torch.autograd.Variable):
            d_loss = d_loss.item()
        if isinstance(logits_loss, torch.autograd.Variable):
            logits_loss = logits_loss.item()
        if isinstance(i_recons_loss, torch.autograd.Variable):
            i_recons_loss = i_recons_loss.item()
        if isinstance(i_part_recons_loss, torch.autograd.Variable):
            i_part_recons_loss = i_part_recons_loss.item()

        wandb.log({
            'g_loss': g_loss,
            'd_loss': d_loss,
            'logits_loss': logits_loss,
            'i_recons_loss': i_recons_loss,
            'i_part_recons_loss': i_part_recons_loss
        })

    def log_fid(self, fid_score):
        """
        Logging fid score
        :param fid_score: ``float``, fid score
        """
        wandb.log({'fid': fid_score})

    def log_image(self, images, num_samples, epoch, batch_idx, normalize=True):
        """
        Create image grid and save it
        :param images: ``Tor    ch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param normalize: if True normalize images
        :param epoch: ``int``, current epoch number
        :param batch_idx: ``int``, current batch number
        """
        nrows = int(np.sqrt(num_samples))
        grid_name = f"epoch{epoch}_step_{batch_idx}.jpg"
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        self.save_torch_images(grid, grid_name)
        wandb.log({'fixed_noise': [wandb.Image(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))]})

    def save_torch_images(self, grid, grid_name):
        """
        Display and save image grid
        :param grid: ``numpy ndarray``, grid image
        :param grid_name: ``str``, grid name for save
        """
        out_dir = self.data_subdir
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        MetricLogger._save_images(fig, out_dir, grid_name)
        plt.close()

    @staticmethod
    def _save_images(fig, out_dir, grid_name):
        """
        Saves image on drive
        :param fig: pls.figure object
        :param out_dir: path to output dir
        :param grid_name: ``str``, grid name for save
        """
        MetricLogger._make_dir(out_dir)
        fig.savefig('{}/{}'.format(out_dir, grid_name))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
