import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DATA_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsDataset(datasets.ImageFolder):
    """Wrapper to make torchvision ImageFolder work with Albumentations"""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path) # PIL Image
        
        if self.transform is not None:
            # Convert to Numpy array for Albumentations
            image = np.array(sample)
            augmented = self.transform(image=image)
            sample = augmented['image']
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def get_dataloaders():
    # Advanced Real-World Augmentation Pipeline
    train_transforms = A.Compose([
        A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=35, p=0.6),
        A.Perspective(scale=(0.05, 0.15), p=0.3),
        A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.2), # Simulates leaf curl
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.7),
        A.RandomShadow(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        A.GaussNoise(var_limit=(5, 30), p=0.25),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    # Validation uses strictly pure images
    val_test_transforms = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
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
