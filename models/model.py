import torch
import torch.nn as nn
from kornia import filter2D


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


class GC(nn.Module):
    """
    Global context
    Based on this work: https://arxiv.org/abs/1904.11492
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.out_channels = int(in_channels * ratio)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax_d2 = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.Conv2d(self.out_channels, in_channels, kernel_size=1)
        )

    def _getcontext(self, x):
        """
        Context modeling block
        :param x: Tensor([N, C, H, W])
        :return: Tensor([N, C, 1, 1])
        """
        N, C, H, W = x.size()
        x_hat = x
        x_hat = x_hat.view(N, C, H * W)     # ℝ[N, C, H, W] --> ℝ[N, C, H * W]
        x_hat = x_hat.unsqueeze(1)          # ℝ[N, C, H * W] --> ℝ[N, 1, C, H * W]
        cm = self.conv_mask(x)              # ℝ[N, 1, C, H * W] --> ℝ[N, 1, H, W]
        cm = cm.view(N, 1, H * W)           # ℝ[N, 1, H, W] --> ℝ[N, 1, H * W]
        cm = self.softmax_d2(cm)            # softmax on second dimension
        cm = cm.unsqueeze(3)                # ℝ[N, 1, H * W] --> ℝ[N, 1, H * W, 1]
        context = torch.matmul(x_hat, cm)   # ℝ[N, 1, C, H * W] ⊗ ℝ[N, 1, H * W, 1] = ℝ[N, 1, C, 1]
        context = context.view(N, C, 1, 1)  # ℝ[N, 1, C, 1] --> ℝ[N, C, 1, 1]
        return context

    def forward(self, x):
        """
        :param x: Tensor([N, C, H, W])
        :return:
        """
        context = self._getcontext(x)           # ℝ[N, C, H, W] --> ℝ[N, C, 1, 1]
        transform = self.transform(context)     # ℝ[N, C, 1, 1]
        fusion = x + transform                  # ℝ[N, C, H, W] ⊕ ℝ[N, C, 1, 1] = ℝ[N, C, H, W]
        return fusion


class Blur(nn.Module):
    """https://richzhang.github.io/antialiased-cnns/"""
    def __init__(self):
        super().__init__()
        kernel = torch.Tensor([1, 2, 1])
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        kernel = self.kernel
        kernel = kernel[None, None, :] * kernel[None, :, None]
        return filter2D(x, kernel, normalized=True)


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            Blur(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        return self.up_block(x)


class Generator(nn.Module):

    def __init__(self, img_size, in_channels=512, img_channels=3, z_dim=256):
        """
        :param img_size: ``2^n``, final resolution
        :param in_channels: ``int``, number of input channels of generator
        :param img_channels: ``int``, 1 for grayscale, 3 for RGB, 4 for transparent
        :param z_dim: ``int``, latent space, in the paper equal 256
        """
        super().__init__()
        assert img_size in [64, 128, 256, 512, 1024], 'image size must be [64, 128, 256, 512, 1024]'
        self.resolution = img_size
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),  # 1x1 to 4x4
            nn.BatchNorm2d(in_channels),
            nn.GLU(dim=1)
        )
        # Upsample blocks
        # input shape [256x4x4]
        self.up_sample_8 = UpSampleBlock(256, 1024)     # output shape ℝ[512x8x8]
        self.up_sample_16 = UpSampleBlock(512, 1024)    # output shape ℝ[512x16x16]
        self.up_sample_32 = UpSampleBlock(512, 512)     # output shape ℝ[256x32x32]
        self.up_sample_64 = UpSampleBlock(256, 256)     # output shape ℝ[128x64x64]
        self.up_sample_128 = UpSampleBlock(128, 128)    # output shape ℝ[64x128x128]
        self.up_sample_256 = UpSampleBlock(64, 64)      # output shape ℝ[32x256x256]
        self.up_sample_512 = UpSampleBlock(32, 6)       # output shape ℝ[3x512x512]
        self.up_sample_1024 = UpSampleBlock(3, 6)       # output shape ℝ[3x1024x1024]

        # Residual blocks
        self.sle_8_to_128 = SLE(512, 64)
        self.sle_16_to_256 = SLE(512, 32)
        self.sle_32_to_512 = SLE(256, 3)

        # Out
        self.output = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # input shape ℝ[N, z_dim, 1, 1] e.g. ℝ[1, 256, 1, 1]
        x_4 = self.initial(x)
        x_8 = self.up_sample_8(x_4)
        x_16 = self.up_sample_16(x_8)
        x_32 = self.up_sample_32(x_16)
        x_64 = self.up_sample_64(x_32)

        x_128 = self.up_sample_128(x_64)
        sle_x_128 = self.sle_8_to_128(x_8, x_128)

        x_256 = self.up_sample_256(sle_x_128)
        sle_x_256 = self.sle_16_to_256(x_16, x_256)

        x_512 = self.up_sample_512(sle_x_256)
        sle_x512 = self.sle_32_to_512(x_32, x_512)

        x_1024 = self.up_sample_1024(sle_x512)
        x = self.output(x_1024)

        return x
