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

__C.GEN_TYPE = 'sle'        # sle or gc
__C.LEARNING_RATE = 1e-3
__C.BATCH_SIZE = 8
# SHAPES
__C.Z_DIMENSION = 256
__C.IMG_SIZE = 1024
__C.CHANNELS_IMG = 3
# Models features
__C.IN_CHANNELS = 512
# Metrics
__C.WANDB_ID = None
# Paths and saves
__C.DATASET_PATH = None
__C.OUT_DIR = ''
__C.PATH_TO_LOG_FILE = 'SLE-GC-GAN/data/logs/train.log'
# Display results
__C.NUM_SAMPLES = 16
__C.FIXED_NOISE_SAMPLES = 16
__C.FREQ = 100
# FID
__C.FID_NUM_SAMPLES = 256
__C.FID_FREQ = 100
# WANDB
__C.RESUME_ID = None
# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(__C.PATH_TO_LOG_FILE, mode='a', encoding='utf-8')

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.setLevel(logging.INFO)
