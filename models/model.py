import torch.nn as nn


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
        # TODO possible need fix or try GlobalContext network
        return self.sle_block(x_low) * x_high


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            # TODO add blur
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GLU(dim=1)
        )

    def forward(self, x):
        return self.up_block(x)


class Generator(nn.Module):

    def __init__(self, img_size, in_channels=512, img_channels=3, z_dim=256):
        """
        :param img_size: ``2^n``, final resolution must be power of 2
        :param in_channels: ``int``, number of input channels of generator
        :param img_channels: ``int``, 1 for grayscale, 3 for RGB, 4 for transparent
        :param z_dim: ``int``, latent space, in the paper equal 256
        """
        super().__init__()
        self.resolution = img_size
        # TODO power of 2 res
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),  # 1x1 to 4x4
            nn.BatchNorm2d(in_channels),
            nn.GLU(dim=1)
        )
        # Upsample blocks
        # input shape [256x4x4]
        self.up_sample_8 = UpSampleBlock(256, 1024)     # output shape [512x8x8]
        self.up_sample_16 = UpSampleBlock(512, 1024)    # output shape [512x16x16]
        self.up_sample_32 = UpSampleBlock(512, 512)     # output shape [256x32x32]
        self.up_sample_64 = UpSampleBlock(256, 256)     # output shape [128x64x64]
        self.up_sample_128 = UpSampleBlock(128, 128)    # output shape [64x128x128]
        self.up_sample_256 = UpSampleBlock(64, 64)      # output shape [32x256x256]
        self.up_sample_512 = UpSampleBlock(32, 6)       # output shape [3x512x512]
        self.up_sample_1024 = UpSampleBlock(3, 6)       # output shape [3x1024x1024]

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
        # input shape [N, z_dim, 1, 1] e.g. [1, 256, 1, 1]
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
