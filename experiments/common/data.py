"""Data loading and augmentation for the ablation.

The augmentation contract (CP2 spec sections 3, 5):

  * The four CP1 "basic" augmentations are applied in BOTH Stack OFF and ON,
    train set only, at identical strengths across every run.
  * Stack ON layers the advanced field-condition pipeline ON TOP of the basic
    four (it does not replace them).
  * Validation and test sets receive resize + ImageNet normalization and
    NOTHING else, ever. A hard assertion enforces this so an accidental
    augmented val/test loader fails loudly instead of silently invalidating
    the generalization measurement.

Because OFF and ON share this single code path, the only augmentation
difference between an OFF/ON pair is whether the advanced block is present —
which is exactly the variable the ablation isolates.
"""
import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import datasets

from .seeding import seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Marker names used by the val/test guard assertion.
_AUGMENTATION_MARKERS = (
    "RandomResizedCrop", "HorizontalFlip", "Rotate", "ColorJitter",
    "Perspective", "ElasticTransform", "RandomShadow", "GaussianBlur",
    "GaussNoise", "ImageCompression", "CoarseDropout", "Affine", "VerticalFlip",
)


class AlbumentationsImageFolder(datasets.ImageFolder):
    """ImageFolder that feeds numpy arrays through an Albumentations transform."""

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            image = np.array(sample)
            sample = self.transform(image=image)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def _basic_four(image_size: int) -> list:
    """The four CP1 augmentations, identical strengths in every run."""
    return [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.5),
    ]


def _advanced_block() -> list:
    """Field-condition augmentations layered on top of the basic four for
    Stack ON only. Moderate strengths chosen to preserve lesion features."""
    return [
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.15),
        A.RandomShadow(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 5), hole_height_range=(1, 15),
                        hole_width_range=(1, 15), p=0.15),
    ]


def build_train_transform(image_size: int, advanced_augmentation: bool) -> A.Compose:
    ops = _basic_four(image_size)
    if advanced_augmentation:
        ops += _advanced_block()
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def build_eval_transform(image_size: int) -> A.Compose:
    """Resize + center-crop + normalize. NO augmentation. Used for val, test,
    and the real-world test set."""
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _assert_no_augmentation(transform: A.Compose, context: str) -> None:
    """Fail loudly if any augmentation op leaked into a val/test transform."""
    names = {type(t).__name__ for t in transform.transforms}
    leaked = names.intersection(_AUGMENTATION_MARKERS)
    if leaked:
        raise AssertionError(
            f"Augmentation {sorted(leaked)} found in {context} transform. "
            "Validation/test must be resize+normalize only (CP2 spec 3.3/7)."
        )


def build_loaders(data_dir: str, image_size: int, batch_size: int,
                  advanced_augmentation: bool, seed: int):
    """Returns (train_loader, val_loader, test_loader, class_to_idx).

    All splits are read from the frozen on-disk ImageFolder layout at
    data_dir/{train,val,test}, which is shared by every run.
    """
    train_tf = build_train_transform(image_size, advanced_augmentation)
    eval_tf = build_eval_transform(image_size)
    _assert_no_augmentation(eval_tf, "val/test")

    train_ds = AlbumentationsImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = AlbumentationsImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds = AlbumentationsImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def build_real_world_loader(real_world_dir: str, image_size: int, batch_size: int):
    """Loader for the held-out real-world test set (eval transform only)."""
    eval_tf = build_eval_transform(image_size)
    _assert_no_augmentation(eval_tf, "real-world test")
    ds = AlbumentationsImageFolder(real_world_dir, transform=eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader, ds.class_to_idx
