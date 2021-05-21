from data.dataset import ImgFolderDataset
from torch.utils.data import DataLoader
from utils import show_batch

if __name__ == '__main__':
    path = ''
    dataset = ImgFolderDataset(path)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=2, shuffle=True)

    batch = next(iter(dataloader))
    show_batch(batch, size=8, shape=(8, 8), save='/home/mirage/')

