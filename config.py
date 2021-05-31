import logging
from easydict import EasyDict as edict

__C = edict()

# for consumers
cfg = __C

# NAMES
__C.PROJECT_NAME = "SLE-GC-GAN"
__C.PROJECT_VERSION_NAME = "SLE-GAN"
__C.DATASET_NAME = ""

# Global
__C.START_EPOCH = 0
__C.END_EPOCH = 1
__C.LEARNING_RATE = 1e-3
__C.BATCH_SIZE = 8
__C.DIFF_AUGMENT_POLICY = 'color,translation,cutout'
# SHAPES
__C.Z_DIMENSION = 256
__C.CHANNELS_IMG = 3
__C.IMG_SIZE = 1024         # final image size
# Models features
__C.IN_CHANNELS = 512
# Metrics
__C.WANDB_ID = None
# Paths and saves
__C.DATASET_PATH = None
__C.OUT_DIR = ''
__C.SAVE = True
__C.SAVE_EPOCH_FREQ = 1
# Display results
__C.NUM_SAMPLES_IMAGES = 16
__C.FIXED_NOISE_SAMPLES = 16
__C.LOG_FREQ = 100
__C.LOG_IMAGE_FREQ = 100
# FID
__C.FID_NUM_SAMPLES = 256   # number of images for fid dataset
__C.FID_FREQ = 100
# WANDB
__C.RESUME_ID = None
# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()

c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
