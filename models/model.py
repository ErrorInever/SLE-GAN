import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.linalg import sqrtm
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


class SLE(nn.Module):
    """
    Skip-layer excitation: transforms low resolution input and
    return tensor product of high resolution input and low resolution input
    For example:
        inputs: low resolution is ℝ^512x8x8, high resolution is ℝ^64x128x128
        tensor produce is y = ℝ^64x128x128 ⊗ F(ℝ^512x8x8, {W_i}), where F is some transformation function
        e.g. transform ℝ^512x8x8 to ℝ^64x1x1; and W_i is model weights to be learned
        output: ℝ^64x128x128
    """
    def __init__(self, low_feature_map, high_feature_map):
        """
        :param low_feature_map: ``int``, low resolution feature map
        :param high_feature_map: ``int``, high resolution feature map
        """
        super().__init__()
        self.intermediate_feature_map = max(3, high_feature_map // 2)  # get intermediate feature map
        self.sle_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Conv2d(low_feature_map, self.intermediate_feature_map, kernel_size=(4, 4)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.intermediate_feature_map, high_feature_map, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x_low, x_high):
        """
        :param x_low: input feature map from low resolution upsample layer
        :param x_high: input feature map from high resolution upsample layer
        :return y: output feature map: y = x_high ⊗ F(x_low, {W_i})
        """
        return self.sle_block(x_low) * x_high


class UpSampleBlock(nn.Module):
    """Upsample block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor(N, C, H, W)``
        """
        return self.up_block(x)


class Generator(nn.Module):
    """Generator"""
    def __init__(self, img_size, in_channels=512, img_channels=3, z_dim=256):
        """
        :param img_size: ``2^n``, final resolution, should be the power of two
        :param in_channels: ``int``, number of input channels of generator
        :param img_channels: ``int``, 1 for grayscale, 3 for RGB, 4 for transparent, default=3
        :param z_dim: ``int``, size of latent space
        """
        super().__init__()
        assert img_size in [256, 512, 1024], 'image size must be [256, 512, 1024]'

        self.img_size = img_size
        self.factors_res = {"256": 32, "512": 3, "1024": 3}
        self.in_features = self.factors_res[str(self.img_size)]
        self.img_channels = img_channels

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1),  # 1x1 to 4x4
            nn.BatchNorm2d(in_channels),
            nn.GLU(dim=1)
        )
        # Upsample blocks
        # input shape [256x4x4]
        self.up_sample_8 = UpSampleBlock(256, 1024)         # output shape ℝ[512x8x8]
        self.up_sample_16 = UpSampleBlock(512, 1024)        # output shape ℝ[512x16x16]
        self.up_sample_32 = UpSampleBlock(512, 512)         # output shape ℝ[256x32x32]
        self.up_sample_64 = UpSampleBlock(256, 256)         # output shape ℝ[128x64x64]
        self.up_sample_128 = UpSampleBlock(128, 128)        # output shape ℝ[64x128x128]

        if img_size >= 256:
            self.up_sample_256 = UpSampleBlock(64, 64)      # output shape ℝ[32x256x256]
        if img_size >= 512:
            self.up_sample_512 = UpSampleBlock(32, 6)       # output shape ℝ[3x512x512]
        if img_size == 1024:
            self.up_sample_1024 = UpSampleBlock(3, 6)       # output shape ℝ[3x1024x1024]

        # SLE blocks
        self.sle_8_to_128 = SLE(512, 64)
        self.sle_16_to_256 = SLE(512, 32)
        if img_size >= 512:
            self.sle_32_to_512 = SLE(256, 3)

        # Out
        self.output = nn.Sequential(
            nn.Conv2d(self.in_features, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, Z, 1, 1])``
        :return: ``Tensor([N, C, H, W])``
        """
        # input shape ℝ[N, z_dim, 1, 1] e.g. ℝ[1, 256, 1, 1]
        x = self.initial(x)

        x_8 = self.up_sample_8(x)
        x_16 = self.up_sample_16(x_8)
        x_32 = self.up_sample_32(x_16)
        x_64 = self.up_sample_64(x_32)
        x_128 = self.up_sample_128(x_64)
        sle_x_128 = self.sle_8_to_128(x_8, x_128)
        x_256 = self.up_sample_256(sle_x_128)
        x = self.sle_16_to_256(x_16, x_256)

        if self.img_size >= 512:
            x_512 = self.up_sample_512(x)
            x = self.sle_32_to_512(x_32, x_512)

            if self.img_size == 1024:
                x = self.up_sample_1024(x)

        x = self.output(x)

        return x


class DownSampleBlock(nn.Module):
    """Down sample block with addition"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample_left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.down_sample_right = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor([N, C, H, W])``
        """
        x_left = self.down_sample_left(x)
        x_right = self.down_sample_right(x)
        return x_left + x_right


class SimpleDecoderBlock(nn.Module):
    """Main block for simple decoders"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.simple_block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        return self.simple_block(x)


class SimpleDecoder(nn.Module):
    """Simple decoder"""
    def __init__(self, in_channels, out_channels=3, num_blocks=4):
        """
        :param num_blocks: ``int``, number blocks
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.body = nn.ModuleList([])

        for i in range(1, num_blocks+1):
            if i != num_blocks:
                self.body.append(SimpleDecoderBlock(in_channels, in_channels))
                in_channels //= 2
            else:
                self.body.append(SimpleDecoderBlock(in_channels, self.out_channels * 2))

    def forward(self, x):
        for layer in self.body:
            x = layer(x)
        # TODO: add Tanh()?
        return x


class InputPartBlock(nn.Module):
    """Block for InputBLockDiscriminator"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.initial = nn.Sequential(
            # TODO: add blur
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor(N, C, H, W)``
        """
        return self.initial(x)


class InputBlockDiscriminator(nn.Module):
    """Input block for Discriminator"""
    def __init__(self, in_features, out_features, factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.initial = nn.ModuleList([])

        if factor != 0:
            for i in range(1, factor + 1):
                if i == 1:
                    self.initial.append(InputPartBlock(in_features, out_features))
                else:
                    self.initial.append(InputPartBlock(out_features, out_features))

        else:
            self.initial.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1),
                nn.LeakyReLU(0.1)
            ))

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``Tensor(N, C, H, W)``
        """
        for layer in self.initial:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    """Discriminator"""
    def __init__(self, img_size, img_channels=3):
        """
        :param img_size: ``2^n``, must be the same as output of the generator
        :param img_channels: ``int``, 1 for grayscale, 3 for RGB, 4 for transparent
        """
        super().__init__()
        assert img_size in [256, 512, 1024], 'image size must be [256, 512, 1024]'

        self.img_size = img_size
        self.img_channels = img_channels
        self.factor_down_sample = {"256": 0, "512": 1, "1024": 2}
        self.ds_factor = self.factor_down_sample[str(img_size)]

        self.initial = InputBlockDiscriminator(img_channels, 16, self.ds_factor)    # output shape ℝ[16,256,256]
        self.down_sample_128 = DownSampleBlock(16, 32)                              # output shape ℝ[32,128,128]
        self.down_sample_64 = DownSampleBlock(32, 64)                               # output shape ℝ[64,64,64]
        self.down_sample_32 = DownSampleBlock(64, 128)                              # output shape ℝ[128,32,32]
        self.down_sample_16 = DownSampleBlock(128, 256)                             # output shape ℝ[256,16,16]
        self.down_sample_8 = DownSampleBlock(256, 512)                              # output shape ℝ[512,8,8]

        self.decoder_part = SimpleDecoder(256)                                      # output shape ℝ[3,128,128]
        self.decoder = SimpleDecoder(512)                                           # output shape ℝ[3,128,128]

        self.real_fake_logits_out = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),  # ℝ[512,8,8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)     # output shape ℝ[1,5,5]
        )

    def forward(self, x):
        """
        :param x: ``Tensor([N, C, H, W])``
        :return: ``List([[N,1,5,5], [N,3,128,128], [N,3,128,128])``
        """
        x = self.initial(x)                                 # output shape ℝ[N, 16, 256, 256]
        x = self.down_sample_128(x)                         # output shape ℝ[N, 32, 128, 128]
        x = self.down_sample_64(x)                          # output shape ℝ[N, 64, 64, 64]
        x = self.down_sample_32(x)                          # output shape ℝ[N, 128, 32, 32]
        x_16 = self.down_sample_16(x)                       # output shape ℝ[N, 256, 16, 16]
        x_8 = self.down_sample_8(x_16)                      # output shape ℝ[N, 512, 8, 8]

        crop_img_8 = center_crop_img(x_16, (8, 8), mode='nearest')     # ℝ[N, 512, 8, 8]

        decoded_img_128_part = self.decoder_part(crop_img_8)            # ℝ[N, 512, 8, 8] --> ℝ[N, 3, 128, 128]
        decoded_img_128 = self.decoder(x_8)                             # ℝ[N, 512, 8, 8] --> ℝ[N, 3, 128, 128]

        real_fake_logits_out = self.real_fake_logits_out(x_8)           # ℝ[N, 512, 8, 8] --> ℝ[1, 5, 5]

        return real_fake_logits_out, decoded_img_128_part, decoded_img_128
