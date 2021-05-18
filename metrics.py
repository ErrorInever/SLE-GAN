import os
import numpy as np
import wandb
import torchvision
import errno
import torch
from matplotlib import pyplot as plt
from config import cfg


class MetricLogger:
    """Metric class"""
    def __init__(self, project_version_name):
        """
        :param project_version_name: name of current version of project
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

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
