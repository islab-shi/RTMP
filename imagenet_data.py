from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_imagenet_data(
    dataset_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Training keeps the standard ImageNet augmentation used by torchvision recipes.
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    # Validation uses deterministic resize and center crop for stable accuracy curves.
    val_transforms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "val"), transform=val_transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataset, train_loader, val_dataset, val_loader
