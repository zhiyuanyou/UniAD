from __future__ import division

import logging
import os.path
import pickle
import random
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger("global_logger")

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_cifar10_dataloader(cfg, training, distributed=True):

    logger.info("building CustomDataset from: {}".format(cfg["root_dir"]))

    dataset = CIFAR10(
        root=cfg["root_dir"],
        train=training,
        resize=cfg["input_size"],
        normals=cfg["normals"],
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool,
        resize: List[int],
        normals: List[int],
    ) -> None:

        self.root = root
        self.normals = normals
        self.train = train  # training set or test set

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize, Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self._select_normal()

    def _select_normal(self) -> None:
        assert self.data.shape[0] == len(self.targets)
        _data_normal = []
        _data_defect = []
        _targets_normal = []
        _targets_defect = []
        for datum, target in zip(self.data, self.targets):
            if target in self.normals:
                _data_normal.append(datum)
                _targets_normal.append(target)
            elif not self.train:
                _data_defect.append(datum)
                _targets_defect.append(target)

        if not self.train:
            ids = random.sample(range(len(_data_defect)), len(_data_normal))
            _data_defect = [_data_defect[idx] for idx in ids]
            _targets_defect = [_targets_defect[idx] for idx in ids]

        self.data = _data_normal + _data_defect
        self.targets = _targets_normal + _targets_defect

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        label = 0 if target in self.normals else 1

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        height = img.shape[1]
        width = img.shape[2]

        if label == 0:
            mask = torch.zeros((1, height, width))
        else:
            mask = torch.ones((1, height, width))

        input = {
            "filename": "{}/{}.jpg".format(classes[target], index),
            "image": img,
            "mask": mask,
            "height": height,
            "width": width,
            "label": label,
            "clsname": "cifar",
        }

        return input

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
