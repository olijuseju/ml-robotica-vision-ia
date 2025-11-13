from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    # Estadísticos estándar para CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tf, test_tf


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Devuelve dataloaders de entrenamiento y test para CIFAR-10.
    num_workers=0 para ir sobre seguro en Windows.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_tf, test_tf = get_transforms()

    train_set = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_set = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_tf,
    )

    pin = True if torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, test_loader
