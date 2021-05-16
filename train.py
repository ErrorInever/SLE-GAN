import argparse
import logging
import os
import time
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from models.model import Generator, Discriminator
from config import cfg
from utils import set_seed
from data.dataset import ImgFolderDataset


def parse_args():
    parser = argparse.ArgumentParser(description='SLE-GC-GAN')
    parser.add_argument('--data_path', dest='data_path', help='path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='path where to save files', default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='path to checkpoint.pth.tar', default=None, type=str)
    parser.add_argument('--device', dest='device', help='use device: gpu, tpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='use this option if you run it from kaggle',
                        default=None, type=str)
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    set_seed(8989)

    logger = logging.getLogger('train')
    args = parse_args()

    assert args.data_path, 'data path not specified'

    if args.api:
        os.environ["WANDB_API_KEY"] = args.api

    if args.wandb_id:
        cfg.WANDB_ID = args.wandb_id

    if args.out_dir:
        cfg.OUT_DIR = args.out_dir

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')
    # elif args.device == 'tpu':
    #     device = xm.xla_device()
    elif args.device is None and not torch.cuda.is_available():
        logger.error(f"device:{args.device}", exc_info=True)
        raise ValueError('Device not specified and gpu is not available')
    logger.info(f'Using device:{args.device}')

    # defining dataset and dataloader
    dataset = ImgFolderDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True,
                            pin_memory=True)
    # defining models
    gen = Generator(img_size=cfg.IMG_SIZE, in_channels=cfg.IN_CHANNELS, img_channels=cfg.CHANNELS_IMG, z_dim=cfg.Z_DIMENSION,
                    res_type=cfg.GEN_TYPE).to(device)
    dis = Discriminator(img_size=cfg.IMG_SIZE, img_channels=cfg.CHANNELS_IMG).to(device)
    # defining optimizers
    opt_gen = optim.Adam(params=gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    opt_dis = optim.Adam(params=dis.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    # defining gradient scalers for automatic mixed precision
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_dis = torch.cuda.amp.GradScaler()

    # TODO: load models and resume training

    gen.train()
    dis.train()

    for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
        pass
