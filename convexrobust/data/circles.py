import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import pytorch_lightning as pl
from sklearn import datasets

class CirclesDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.split = [0.5, 0.3, 0.2]
        self.batch_size = 30
        self.in_n = 2

    def setup(self, stage=None):
        X, y = datasets.make_circles(n_samples=400, noise=0.1, factor=0.2)
        X *= 0.7

        # all_data = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
        all_data = TensorDataset(torch.tensor(X).float(), 1 - torch.tensor(y).long())

        train_size = int(self.split[0] * len(all_data))
        val_size = int(self.split[1] * len(all_data))
        test_size = len(all_data) - train_size - val_size

        self.train_data, self.val_data, self.test_data = \
            random_split(all_data, [train_size, val_size, test_size])

        self.dataset_test = self.test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0)
