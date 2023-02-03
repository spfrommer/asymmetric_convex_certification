from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datasets import TrialCIFAR10

from typing import Optional, Sequence, Any

from convexrobust.utils import dirs

class CIFAR10SelectDataModule(CIFAR10DataModule):
    dataset_cls = TrialCIFAR10
    dims = (3, 32, 32)

    def __init__(
        self,
        val_split: float = 0.2,
        num_workers: int = 0,
        labels: Optional[Sequence] = (1, 5, 8),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dirs.data_path('cifar10'), val_split, num_workers, *args, **kwargs)

        self.labels = labels if labels is not None else set(range(10))
        self.EXTRA_ARGS = dict(labels=self.labels, relabel=True, num_samples=6000)

    @property
    def num_classes(self) -> int:
        return len(self.labels)
