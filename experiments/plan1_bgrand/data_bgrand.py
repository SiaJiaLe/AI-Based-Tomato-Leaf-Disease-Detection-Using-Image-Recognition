"""Train loader with background randomization prepended (Plan 1).

Mirrors experiments.common.data.build_loaders but inserts BackgroundRandomize
at the FRONT of the train transform chain. Val/test are built with the SHARED
eval transform from common.data, so they are guaranteed identical to every
other run (resize + centre-crop + normalize, no augmentation ever).

The only difference from the ablation's train pipeline is the one extra
transform — exactly the variable Plan 1 isolates.
"""
import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from experiments.common.data import (AlbumentationsImageFolder, IMAGENET_MEAN,
                                      IMAGENET_STD, _advanced_block, _basic_four,
                                      build_eval_transform)
from experiments.common.seeding import seed_worker

from .bg_randomize import BackgroundRandomize


def build_train_transform_bgrand(image_size: int, advanced_augmentation: bool,
                                 bg_cfg: dict) -> A.Compose:
    """basic-4 (+ advanced if ON) preceded by background randomization."""
    ops = [BackgroundRandomize(
        background_dir=bg_cfg["background_dir"],
        prob=bg_cfg.get("prob", 0.5),
        boundary_blur=bg_cfg.get("boundary_blur", True),
    )]
    ops += _basic_four(image_size)
    if advanced_augmentation:
        ops += _advanced_block()
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def build_loaders_bgrand(data_dir: str, image_size: int, batch_size: int,
                         advanced_augmentation: bool, seed: int, bg_cfg: dict):
    """Returns (train_loader, val_loader, class_to_idx).

    train  -> background-randomized pipeline
    val    -> shared eval transform (NO augmentation), identical to the ablation
    """
    train_tf = build_train_transform_bgrand(image_size, advanced_augmentation, bg_cfg)
    eval_tf = build_eval_transform(image_size)  # shared, augmentation-free

    train_ds = AlbumentationsImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = AlbumentationsImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.class_to_idx
