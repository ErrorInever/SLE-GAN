import argparse
import logging
import os
import time
import torch
import torch.optim as optim
from kornia.geometry.transform import resize
from kornia.geometry.transform.crop.crop2d import center_crop
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model import Generator, Discriminator, InceptionV3FID
from config import cfg
from utils import set_seed, save_checkpoint, load_checkpoint, get_random_noise, print_epoch_time, get_sample_dataloader
from data.dataset import ImgFolderDataset, FIDNoiseDataset
from losses import reconstruction_loss_mse, hinge_loss
from metrics import MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description='SLE-GC-GAN')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='Path to checkpoint.pth.tar', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu, tpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--diff_aug', dest='diff_aug', help='Use differentiable augmentation', action='store_true')
    parser.add_argument('--fid', dest='fid', help='Display fid score', action='store_true')
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle, '
                                                              'input api key', default=None, type=str)
    parser.print_help()
    return parser.parse_args()


@print_epoch_time
def train_one_epoch(gen, opt_gen, scaler_gen, dis, opt_dis, scaler_dis, dataloader, metric_logger, device,
                    fixed_noise, epoch, fid_model, fid_score):
    """
    Train one epoch
    :param gen: ``Instance of models.model.Generator``, generator model
    :param opt_gen: ``Instance of torch.optim``, optimizer for generator
    :param scaler_gen: ``torch.cuda.amp.GradScaler()``, gradient scaler for generator
    :param dis: ``Instance of models.model.Discriminator``, discriminator model
    :param opt_dis: ``Instance of torch.optim``, optimizer for discriminator
    :param scaler_dis: ``torch.cuda.amp.GradScaler()``, gradient scaler for discriminator
    :param dataloader: ``Instance of torch.utils.data.DataLoader``, train dataloader
    :param metric_logger: ``Instance of metrics.MetricLogger``, logger
    :param device: ``Instance of torch.device``, cuda device
    :param fixed_noise: ``Tensor([N, 1, Z, 1, 1])``, fixed noise for display result
    :param epoch: ``int``, current epoch
    :param fid_model: model for calculate fid score
    :param fid_score: ``bool``, if True - calculate fid score otherwise no
    """
    loop = tqdm(dataloader, leave=True)
    for batch_idx, real in enumerate(loop):
        cur_batch_size = real.shape[0]
        real = real.to(device)
        real.requires_grad_()
        real_cropped_128 = center_crop(real, size=(128, 128))
        real_128 = resize(real, size=(128, 128))
        noise = torch.randn(cur_batch_size, cfg.Z_DIMENSION, 1, 1).to(device)
        # Train discriminator
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                fake = gen(noise)
            real_fake_logits_real_images, decoded_real_img_part, decoded_real_img = dis(real)
            real_fake_logits_fake_images, _, _ = dis(fake.detach())
            # maximize divergence between real and fake data
            # TODO: try dual contrastive loss instead simple hinge
            divergence = hinge_loss(real_fake_logits_real_images, real_fake_logits_fake_images)
            i_recon_loss = reconstruction_loss_mse(real_128, decoded_real_img)
            i_part_recon_loss = reconstruction_loss_mse(real_cropped_128, decoded_real_img_part)
            d_loss = divergence + i_recon_loss + i_part_recon_loss

        opt_dis.zero_grad()
        scaler_dis.scale(d_loss).backward()
        scaler_dis.step(opt_dis)
        scaler_dis.update()

        opt_gen.zero_grad()
        noise = torch.randn(cur_batch_size, cfg.Z_DIMENSION, 1, 1).to(device)
        # Train generator
        with torch.cuda.amp.autocast():
            # maximize E[D(G(z))], also we can minimize the negative of that
            fake = gen(noise)
            fake_logits, _, _ = dis(fake)
            g_loss = torch.mean(fake_logits)

        scaler_gen.scale(g_loss).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        # Eval and metrics
        if fid_score:
            if epoch % cfg.FID_FREQ == 0:
                # TODO: test fid on GPU
                fid = evaluate(gen, fid_model, device)
                gen.train()
                metric_logger.log_fid(fid)
        if batch_idx % cfg.LOG_FREQ == 0:
            metric_logger.log(g_loss, d_loss, divergence, i_recon_loss, i_part_recon_loss)
        if batch_idx % cfg.LOG_IMAGE_FREQ == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise)
                metric_logger.log_image(fixed_fakes, cfg.NUM_SAMPLES_IMAGES, epoch, batch_idx, normalize=True)

        loop.set_postfix(
            d_loss=d_loss.item(),
            g_loss=g_loss.item()
        )


@torch.no_grad()
def evaluate(gen, fid_model, device):
    """
    Fid evaluate
    :param gen: Generator
    :param fid_model: Inception_v3 model
    :param device: torch device
    :return: ``float``, fid score
    """
    gen.eval()
    real_dataset = ImgFolderDataset(cfg.DATASET_PATH, fid=True)
    real_dataloader = get_sample_dataloader(real_dataset, num_samples=cfg.FID_NUM_SAMPLES,
                                            batch_size=cfg.BATCH_SIZE)
    noise = torch.randn([len(real_dataloader), cfg.Z_DIMENSION, 1, 1])
    fake_images = []

    for batch in noise:
        batch = batch.to(device)
        fake_images.append(gen(batch.unsqueeze(0)))

    noise_dataset = FIDNoiseDataset(fake_images)
    fake_dataloader = DataLoader(noise_dataset, batch_size=cfg.BATCH_SIZE)

    fid = fid_model.get_fid_score(real_dataloader, fake_dataloader)

    return fid


if __name__ == '__main__':
    set_seed(8989)

    logger = logging.getLogger('train')
    args = parse_args()

    assert args.data_path, 'data path not specified'

    cfg.DATASET_PATH = args.data_path

    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    if args.wandb_id:
        cfg.WANDB_ID = args.wandb_id

    if args.out_dir:
        cfg.OUT_DIR = args.out_dir

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')
    # TODO: device for TPU
    elif args.device is None and not torch.cuda.is_available():
        logger.error(f"device:{args.device}", exc_info=True)
        raise ValueError('Device not specified and gpu is not available')
    logger.info(f'Using device:{args.device}')

    if args.fid:
        fid_model = InceptionV3FID(device)
    else:
        fid_model = None

    # defining dataset and dataloader
    dataset = ImgFolderDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True,
                            pin_memory=True)
    # defining models
    gen = Generator(img_size=cfg.IMG_SIZE, in_channels=cfg.IN_CHANNELS, img_channels=cfg.CHANNELS_IMG,
                    z_dim=cfg.Z_DIMENSION).to(device)
    dis = Discriminator(img_size=cfg.IMG_SIZE, img_channels=cfg.CHANNELS_IMG, diff_aug=args.diff_aug).to(device)
    # defining optimizers
    opt_gen = optim.Adam(params=gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    opt_dis = optim.Adam(params=dis.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    # defining gradient scalers for automatic mixed precision
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_dis = torch.cuda.amp.GradScaler()

    if args.checkpoint:
        fixed_noise, cfg.START_EPOCH = load_checkpoint(args.checkpoint, gen, opt_gen, scaler_gen,
                                                       dis, opt_dis, scaler_dis, cfg.LEARNING_RATE)
    else:
        fixed_noise = get_random_noise(cfg.FIXED_NOISE_SAMPLES, cfg.Z_DIMENSION, device)

    gen.train()
    dis.train()

    metric_logger = MetricLogger(cfg.PROJECT_VERSION_NAME)

    for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
        train_one_epoch(gen, opt_gen, scaler_gen, dis, opt_dis, scaler_dis, dataloader, metric_logger, device,
                        fixed_noise, epoch, fid_model, fid_score=args.fid)
        if cfg.SAVE:
            if epoch % cfg.SAVE_EPOCH_FREQ == 0:
                save_checkpoint(gen, opt_gen, scaler_gen, dis, opt_dis, scaler_dis, fixed_noise, epoch)
