"""
Fixed CNN feature extraction using a pretrained ResNet34 backbone
(ImageNet weights, no fine-tuning, fc layer removed).

This is the standard approach used in existing literature for combining
classical ML classifiers with CNN feature extraction (e.g. Khan et al.
2021; Tan et al. 2021, both cited in the FYP literature review): the
backbone is frozen and only provides a 512-dim descriptor per image, the
classical classifier does all of the class-discriminative learning.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE


class AlbumentationsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            image = np.array(sample)
            sample = self.transform(image=image)['image']
        return sample, target


def _eval_transform():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_feature_extractor(device):
    """Pretrained ResNet34, ImageNet weights, no fine-tuning, fc removed."""
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def extract_folder(folder_path, device):
    """Returns (features [N, 512], labels [N], class_to_idx) for any
    ImageFolder-structured directory (used for both the processed
    train/test splits and the real-world test set)."""
    dataset = AlbumentationsDataset(folder_path, transform=_eval_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    extractor = build_feature_extractor(device)

    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            feats = extractor(inputs)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels), dataset.class_to_idx


def extract_split(split_name, device):
    """Returns (features [N, 512], labels [N], class_to_idx) for a
    named split (e.g. "train", "test") under DATA_DIR."""
    return extract_folder(os.path.join(DATA_DIR, split_name), device)
