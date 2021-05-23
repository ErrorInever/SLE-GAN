import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.linalg import sqrtm
from kornia import filter2D
from utils import center_crop_img
from torchvision.models.inception import inception_v3


class InceptionV3FID(nn.Module):
    """Inception_v3 for calculate FID metric"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = inception_v3(pretrained=True, progress=True, transform_input=True).to(self.device)
        self.fc = self.model.fc
        self.model.fc = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)
        self.model.eval()

    def forward(self, x):
        """
        :param x: ``Tensor([N, 3, 299, 299]), in range[-1, 1]``
        """
        # features map
        x = self.model(x)
        features = x.data.cpu().numpy()
        # prediction
        x = self.fc(x)
        #   prob = self.softmax(x)
        return features

    @staticmethod
    def _calculate_stats(feature_map):
        mu = np.mean(feature_map, axis=0)
        sigma = np.cov(feature_map, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2):
        """
        Calculate frechet inception distance
        :param mu1: the feature-wise mean of the real data
        :param sigma1: covariance matrix for the real data
        :param mu2: the feature-wise mean of the generated data
        :param sigma2: covariance matrix for the generated
        :return:
        """
        ssdif = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdif + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def _get_encoded_stats(self, dataloader, num_classes=1000):
        """
        Get encoded stats and probabilities from pool3_ft layer of inception_v3
        :param dataloader: ``Instance of torch.utils.data.DataLoader``, dataloader
        :param num_classes: ``int``, number of classes of classifier, default = 1000
        """
        num_img = len(dataloader.dataset)
        feature_map = np.zeros((num_img, 2048))
        # predictions = np.zeros((num_img, num_classes))
        loop = tqdm(dataloader, leave=True)
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(self.device)
            batch_size = batch.shape[0]
            i = batch_idx * batch_size
            j = i + batch_size
            with torch.no_grad():
                feature_map[i:j] = self.forward(batch)

        return feature_map

    def get_fid_score(self, real_dataloader, fake_dataloader):
        """
        get fid score
        :param real_dataloader: ``torch.data.Dataloader``, dataloader of real data
        :param fake_dataloader: ``torch.data.Dataloader``, dataloader of fake data
        :return: ``float``, fid score
        """
        real_encodes = self._get_encoded_stats(real_dataloader)
        fake_encodes = self._get_encoded_stats(fake_dataloader)

        real_mu, real_sigma = self._calculate_stats(real_encodes)
        fake_mu, fake_sigma = self._calculate_stats(fake_encodes)

        fid = self._calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)

        return fid