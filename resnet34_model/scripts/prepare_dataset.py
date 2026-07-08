import os
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def balance_and_split():
    if not RAW_DIR.exists():
        print(f"Error: {RAW_DIR} does not exist.")
        return

    # 1. Count images per class
    class_counts = {}
    for class_dir in RAW_DIR.iterdir():
        if class_dir.is_dir():
            images = [f for f in class_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            class_counts[class_dir.name] = len(images)

    if not class_counts:
        print("No classes found in raw directory.")
        return

    print("--- Current Image Counts ---")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    # 2. Find total images per class (no balancing)
    print("\nKeeping all images (no balancing).")

    # 3. Clean only train, val, and test directories (so we don't delete real_environment_test)
    import shutil
    for split in ['train', 'val', 'test']:
        split_dir = PROCESSED_DIR / split
        if split_dir.exists():
            print(f"Cleaning up old {split} directory...")
            shutil.rmtree(split_dir)

    # Create processed structure
    for split in ['train', 'val', 'test']:
        for cls in class_counts.keys():
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    symlink_count = 0

    # 4. Process each class
    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        images = [f for f in class_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(images)

        class_total = len(images)
        
        # Calculate splits for ALL images in this class
        train_end = int(class_total * TRAIN_RATIO)
        val_end = train_end + int(class_total * VAL_RATIO)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Create SYMLINKS instead of copying! Takes 0 bytes!
        for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_cls_dir = PROCESSED_DIR / split / class_dir.name
            for img in img_list:
                symlink_path = split_cls_dir / img.name
                # Create relative symlink
                os.symlink(img, symlink_path)
                symlink_count += 1

    print("\n--- Summary ---")
    print(f"Created Train/Val/Test splits via SYMLINKS for ALL images: {symlink_count} files linked (0 bytes of extra disk space used!).")
    print("\nDataset is split and ready for training!")

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    balance_and_split()
