import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import cfg


class ImgFolderDataset(Dataset):
    """Custom dataset from folder where stored images"""
    def __init__(self, img_folder_path, fid=False):
        """
        :param img_folder_path: ``str``, path to folder where stored images
        """
        self.fid = fid
        self.img_folder = img_folder_path
        self.img_names = [n for n in os.listdir(img_folder_path) if n.endswith(('png', 'jpeg', 'jpg'))]

    def __getitem__(self, idx):
        if cfg.CHANNELS_IMG == 3:
            img = Image.open(os.path.join(self.img_folder, self.img_names[idx])).convert('RGB')
        else:
            img = Image.open(os.path.join(self.img_folder, self.img_names[idx]))

        if self.fid:
            return self.fid_transform(img)
        else:
            return self.transform(img)

    def __len__(self):
        return len(self.img_names)

    @property
    def transform(self):
        """
        Resizes images to resolution specified in config, does random horizontal flip with probability 0.5, convert to
        tensor and normalize
        """
        return transforms.Compose([
            transforms.Resize(cfg.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5 for _ in range(cfg.CHANNELS_IMG)],
                                 std=[0.5 for _ in range(cfg.CHANNELS_IMG)])
            ])

    @property
    def fid_transform(self):
        """
        Default transform for inception_v3 fid
        """
        return transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
