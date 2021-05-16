import unittest
import torch
from models.model import Generator, Discriminator


class TestModels(unittest.TestCase):

    def setUp(self):
        self.img_size = 1024
        self.in_channels = 512
        self.z_dim = 256
        self.res_type = ['sle', 'gc']

    def test_generator_shape(self):
        gen_sle = Generator(img_size=self.img_size, in_channels=self.in_channels, z_dim=self.z_dim,
                            res_type=self.res_type[0])
        gen_gc = Generator(img_size=self.img_size, in_channels=self.in_channels, z_dim=self.z_dim,
                           res_type=self.res_type[1])

        noise = torch.randn([1, 256, 1, 1])

        z_gen_sle = gen_sle(noise)
        z_gen_gc = gen_gc(noise)

        real_shape = (1, 3, self.img_size, self.img_size)

        self.assertEqual(z_gen_sle.shape, z_gen_gc.shape)
        self.assertEqual(z_gen_sle.shape, real_shape)
        self.assertEqual(z_gen_gc.shape, real_shape)

    def test_discriminator_shape(self):
        disc = Discriminator(img_size=self.img_size)

        img = torch.randn([1, 3, 1024, 1024])

        logits_shape = (1, 1, 5, 5)
        decode_img_part = (1, 3, 128, 128)
        decoded_img = (1, 3, 128, 128)

        real_fake_logits_out, decoded_img_128_part, decoded_img_128 = disc(img)

        self.assertEqual(real_fake_logits_out.shape, logits_shape)
        self.assertEqual(decoded_img_128_part.shape, decoded_img_128.shape)
        self.assertEqual(decoded_img_128_part.shape, decode_img_part)
        self.assertEqual(decoded_img_128.shape, decoded_img)
