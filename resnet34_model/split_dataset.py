import os
import shutil
import random

# Set paths
base_dir = os.path.expanduser("~/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition/resnet34_model/data")
raw_dir = os.path.join(base_dir, "raw")
processed_dir = os.path.join(base_dir, "processed")

# Split ratios (70% Train, 15% Validation, 15% Test)
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# 1. Create the processed/train, processed/val, processed/test directories
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(processed_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

# 2. Get list of all disease classes
classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
print(f"Found {len(classes)} classes. Starting split (70/15/15)...\n")

total_copied = 0

for cls in classes:
    class_raw_dir = os.path.join(raw_dir, cls)
    images = [f for f in os.listdir(class_raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    # Shuffle images to ensure random distribution
    random.seed(42) # Set seed for reproducibility
    random.shuffle(images)
    
    # Calculate exact numbers for splits
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    # Split the lists
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    # 3. Create the class folders inside train/val/test and copy images
    for split_name, split_images in splits.items():
        split_class_dir = os.path.join(processed_dir, split_name, cls)
        if not os.path.exists(split_class_dir):
            os.makedirs(split_class_dir)
            
        for img in split_images:
            src = os.path.join(class_raw_dir, img)
            dst = os.path.join(split_class_dir, img)
            # Using copy2 preserves file metadata. 
            shutil.copy2(src, dst)
            total_copied += 1
            
    print(f"{cls:<40} -> Train: {len(train_imgs):<5} | Val: {len(val_imgs):<5} | Test: {len(test_imgs):<5}")

print(f"\nSuccessfully split {total_copied} images into the processed folder!")
