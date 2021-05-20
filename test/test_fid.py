import unittest
import torch
from models.model import InceptionV3FID, Generator
from data.dataset import ImgFolderDataset, FIDNoiseDataset
from utils import get_sample_dataloader
from torch.utils.data import DataLoader


class TestFID(unittest.TestCase):

    def setUp(self):
        pass

    def test_fid(self):
        real = torch.zeros((4, 3, 299, 299))
        fake_data = torch.ones((4, 3, 299, 299))
        fid_model = InceptionV3FID(torch.device('cpu'))
        fid_score = fid_model.get_fid_score(real, fake_data)
        print(fid_score)

    def test_fid_with_images(self):
        gen = Generator(1024)
        fid_model = InceptionV3FID(torch.device('cpu'))

        real_dataset = ImgFolderDataset('', fid=True)
        real_dataloader = get_sample_dataloader(real_dataset, num_samples=4,
                                                batch_size=2)

        noise = torch.randn([len(real_dataloader), 256, 1, 1])
        fake_images = []
        for batch in noise:
            fake_images.append(gen(batch.unsqueeze(0)))

        noise_dataset = FIDNoiseDataset(fake_images)
        fake_dataloader = DataLoader(noise_dataset, batch_size=2)

        fid = fid_model.get_fid_score(real_dataloader, fake_dataloader)

        print(fid)


if __name__ == '__main__':
    unittest.main()
