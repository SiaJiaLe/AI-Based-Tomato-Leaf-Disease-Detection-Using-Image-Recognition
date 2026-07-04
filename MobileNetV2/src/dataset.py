import os
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDataset(datasets.ImageFolder):
    """Wrapper to make torchvision ImageFolder work with Albumentations"""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)  # PIL Image

        if self.transform is not None:
            image = np.array(sample)
            augmented = self.transform(image=image)
            sample = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_dataloaders():
    """
    Standard baseline augmentation: basic geometric transforms only
    (random-resized crop, horizontal flip, small rotation). This is what
    existing literature applies for this architecture — it deliberately
    does NOT use the proposed ResNet34 model's advanced field-condition
    augmentation pipeline (shadows, elastic transform, blur, noise, etc.).
    """
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    val_test_transforms = A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    train_dataset = AlbumentationsDataset(train_dir, transform=train_transforms)
    val_dataset = AlbumentationsDataset(val_dir, transform=val_test_transforms)
    test_dataset = AlbumentationsDataset(test_dir, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
