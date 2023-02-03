from pl_bolts.datamodules import FashionMNISTDataModule
from torchvision.datasets import FashionMNIST

from functools import reduce
from operator import ior

from typing import Optional, Sequence, Any

from convexrobust.utils import dirs

class FashionMNISTSelectDataModule(FashionMNISTDataModule):
    def __init__(
        self,
        val_split: float = 0.2,
        num_workers: int = 0,
        labels: Optional[Sequence] = (1, 5, 8),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dirs.data_path('fashion_mnist'), val_split, num_workers, *args, **kwargs)

        self.labels = labels if labels is not None else set(range(10))

    def setup(self, stage=None):
        super().setup(stage=stage)

        train_ts_empty = self.train_transforms is None
        val_ts_empty = self.val_transforms is None
        test_ts_empty = self.test_transforms is None

        train_ts = self.default_transforms() if train_ts_empty else self.train_transforms
        val_ts = self.default_transforms() if val_ts_empty else self.val_transforms
        test_ts = self.default_transforms() if test_ts_empty else self.test_transforms

        dataset_train = self.select_labels(FashionMNIST(self.data_dir, train=True, transform=train_ts))
        dataset_val = self.select_labels(FashionMNIST(self.data_dir, train=True, transform=val_ts))

        self.dataset_train = self._split_dataset(dataset_train)
        self.dataset_val = self._split_dataset(dataset_val, train=False)

        self.dataset_test = self.select_labels(FashionMNIST(self.data_dir, train=False, transform=test_ts))

    def select_labels(self, dataset_full):
        idx = reduce(ior, [dataset_full.targets == i for i in self.labels])
        dataset_full.targets = dataset_full.targets[idx]
        dataset_full.data = dataset_full.data[idx]

        for i, target in enumerate(dataset_full.targets):
            dataset_full.targets[i] = self.labels.index(target)

        return dataset_full

    @property
    def num_classes(self) -> int:
        return len(self.labels)
