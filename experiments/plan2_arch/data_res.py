"""Resolution-aware loaders for Plan 2 Tier 2.

`common.data.build_loaders` already takes an `image_size`, but its eval
transform hardcodes `Resize(256,256)` -> `CenterCrop(image_size)`. At the
baseline's 224 that is a 224/256 = 0.875 crop. Passing 240 through it unchanged
would give a 240/256 = 0.9375 crop — i.e. a WIDER FIELD OF VIEW as well as more
pixels. That is two variables, and the row would no longer isolate resolution.

So the eval resize target scales with the resolution: `resize_to =
round(image_size / 0.875)` (274 for 240), keeping FOV at 240/274 = 0.876 ~=
0.875. The row then tests exactly what plan2 Tier 2 claims — more spatial detail
for the same framing — rather than "sees more of the leaf".

The train side needs no special handling and is reused as-is: `_basic_four` uses
`RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0))`, which
draws the same FOV distribution at any output size.

Everything else (the basic four, the advanced block, the ImageFolder wrapper,
the normalization constants, the no-augmentation guard, the worker seeding) is
imported from `common.data` unchanged.
"""
import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from experiments.common.data import (IMAGENET_MEAN, IMAGENET_STD,
                                     AlbumentationsImageFolder, _advanced_block,
                                     _assert_no_augmentation, _basic_four)
from experiments.common.seeding import seed_worker

# The baseline's crop ratio: 224/256. Held constant so resolution is the only
# thing that changes.
BASELINE_CROP_RATIO = 224 / 256


def default_resize_to(image_size: int) -> int:
    """Resize target that preserves the baseline's 0.875 crop ratio."""
    return int(round(image_size / BASELINE_CROP_RATIO))


def build_train_transform_res(image_size: int, advanced_augmentation: bool) -> A.Compose:
    """Identical to common.data.build_train_transform — restated only so the
    resolution flows through explicitly."""
    ops = _basic_four(image_size)
    if advanced_augmentation:
        ops += _advanced_block()
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def build_eval_transform_res(image_size: int, resize_to: int = None) -> A.Compose:
    """Resize + center-crop + normalize at the run's own resolution. NO
    augmentation — used for val, test, and real-world alike."""
    resize_to = default_resize_to(image_size) if resize_to is None else resize_to
    if resize_to < image_size:
        raise ValueError(f"resize_to ({resize_to}) < image_size ({image_size}): "
                         "CenterCrop would upscale-crop and change framing.")
    return A.Compose([
        A.Resize(height=resize_to, width=resize_to),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_loaders_res(data_dir: str, image_size: int, batch_size: int,
                      advanced_augmentation: bool, seed: int, resize_to: int = None):
    """Same contract as common.data.build_loaders, at an arbitrary resolution.

    Returns (train_loader, val_loader, test_loader, class_to_idx) from the same
    frozen on-disk layout at data_dir/{train,val,test}.
    """
    resize_to = default_resize_to(image_size) if resize_to is None else resize_to
    train_tf = build_train_transform_res(image_size, advanced_augmentation)
    eval_tf = build_eval_transform_res(image_size, resize_to)
    _assert_no_augmentation(eval_tf, "val/test")  # same guard as the baseline

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
    print(f"Loaders at {image_size}px (Resize {resize_to} -> CenterCrop {image_size}, "
          f"FOV {image_size/resize_to:.3f} vs baseline {BASELINE_CROP_RATIO:.3f}).", flush=True)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx
