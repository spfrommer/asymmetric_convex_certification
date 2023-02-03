from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import glob
from PIL import Image
import itertools
import urllib.request
from torchvision.datasets.utils import extract_archive

from convexrobust.utils import dirs

malware_names = [
    'Allaple.L', 'Yuner.A', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3',
    'C2LOP.P', 'C2LOP.gen!g', 'Instantaccess', 'Swizzor.gen!I', 'Swizzor.gen!E',
    'VB.AT', 'Fakerean', 'Alueron.gen!J', 'Malex.gen!J', 'Lolyda.AT', 'Adialer.C',
    'Wintrim.BX', 'Dialplatform.B', 'Dontovo.A', 'Obfuscator.AD', 'Agent.FYI',
    'Autorun.K', 'Rbot!gen', 'Skintrim.N', 'Allaple.A'
]

class MalimgDataset(Dataset):
    def __init__(self, class_0_files, class_1_files, data_transform, stage='train', binarize=True):
        # Class 0 is 24 types of malware, class 1 is allaple.A
        self.file_list = class_0_files + class_1_files
        self.class_cutoff = len(class_0_files)

        self.idx_lookup = np.random.permutation(len(self.file_list))  # Shuffle data

        self.data_transform = data_transform
        self.stage = stage
        self.binarize = binarize # sometimes, we want to return full class label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[self.idx_lookup[idx]]
        img = Image.open(img_path)

        img_transformed = self.data_transform[self.stage](img)

        if self.binarize:
            label = int(self.idx_lookup[idx] >= self.class_cutoff)
        else:
            malware_name = self.file_list[self.idx_lookup[idx]].split('/')[-2]
            try:
                label = malware_names.index(malware_name)
            except:
                print(f'Could not find malware: {malware_name}')

        return img_transformed, label


class MalimgDataModule(pl.LightningDataModule):
    def __init__(self, train_transforms, val_transforms, test_transforms,
                 batch_size=64, num_workers=4, shuffle=True, drop_last=False, binarize=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.binarize = binarize

        self.data_trans = {
            'train': train_transforms, 'val': val_transforms, 'test': test_transforms
        }

        self.seed = 42

    def prepare_data(self):
        if not os.path.isdir(dirs.data_path('malimg')):
            print('Downloading malimg dataset...')
            url = 'https://www.dropbox.com/s/ep8qjakfwh1rzk4/malimg_dataset.zip?dl=1'
            urllib.request.urlretrieve(url, dirs.data_path('malimg.zip'))
            print('Extracting...')
            extract_archive(dirs.data_path('malimg.zip'), dirs.data_path(), True)
            os.rename(dirs.data_path('malimg_paper_dataset_imgs'), dirs.data_path('malimg'))

    def setup(self, stage=None):
        malware_names = glob.glob(dirs.data_path('malimg', '*'))
        malware_names = [n.split('/')[-1] for n in malware_names]
        class_1_names = ['Allaple.A']
        class_0_names = [n for n in malware_names if n not in class_1_names]  # Certifying

        class_0_files = self.get_all_files(class_0_names)
        class_1_files = self.get_all_files(class_1_names)

        if len(class_1_files) > len(class_0_files):
            class_1_files = random.sample(class_1_files, len(class_0_files))
        elif len(class_0_files) > len(class_1_files):
            class_0_files = random.sample(class_0_files, len(class_1_files))

        class_0_train, class_0_test = train_test_split(
            class_0_files, test_size=0.2, random_state=self.seed
        )
        class_0_train, class_0_val = train_test_split(
            class_0_train, test_size=0.2, random_state=self.seed
        )

        class_1_train, class_1_test = train_test_split(
            class_1_files, test_size=0.2, random_state=self.seed
        )
        class_1_train, class_1_val = train_test_split(
            class_1_train, test_size=0.2, random_state=self.seed
        )

        self.dataset_train = MalimgDataset(
            class_0_train, class_1_train, self.data_trans, stage='train', binarize=self.binarize
        )
        self.dataset_val = MalimgDataset(
            class_0_val, class_1_val, self.data_trans, stage='val', binarize=self.binarize
        )
        self.dataset_test = MalimgDataset(
            class_0_test, class_1_test, self.data_trans, stage='test', binarize=self.binarize
        )

    def get_all_files(self, class_names):
        class_files = [glob.glob(dirs.data_path('malimg', n, '*.png')) for n in class_names]
        return list(itertools.chain(*class_files))

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=self.shuffle, drop_last=self.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, num_workers=self.num_workers)
