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

    # 2. Find minimum class count for balancing
    min_count = min(class_counts.values())
    print(f"\nMinimum images in a class: {min_count}")
    print(f"Balancing all classes to exactly {min_count} images to prevent bias AND save disk space...\n")

    # 3. Clean processed directory if it exists (so we can start fresh)
    import shutil
    if PROCESSED_DIR.exists():
        print("Cleaning up old processed directory to free space...")
        shutil.rmtree(PROCESSED_DIR)

    # Create processed structure
    for split in ['train', 'val', 'test']:
        for cls in class_counts.keys():
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    deleted_count = 0
    symlink_count = 0

    # 4. Process each class
    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        images = [f for f in class_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(images)

        # Keep only 'min_count' images, delete the rest to permanently free up space!
        images_to_keep = images[:min_count]
        images_to_delete = images[min_count:]

        for img in images_to_delete:
            img.unlink()
            deleted_count += 1

        # Calculate splits for the kept images
        train_end = int(min_count * TRAIN_RATIO)
        val_end = train_end + int(min_count * VAL_RATIO)

        train_imgs = images_to_keep[:train_end]
        val_imgs = images_to_keep[train_end:val_end]
        test_imgs = images_to_keep[val_end:]

        # Create SYMLINKS instead of copying! Takes 0 bytes!
        for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_cls_dir = PROCESSED_DIR / split / class_dir.name
            for img in img_list:
                symlink_path = split_cls_dir / img.name
                # Create relative symlink
                os.symlink(img, symlink_path)
                symlink_count += 1

    print("\n--- Summary ---")
    print(f"Deleted excess images to balance dataset: {deleted_count} files removed.")
    print(f"Created Train/Val/Test splits via SYMLINKS: {symlink_count} files linked (0 bytes of extra disk space used!).")
    print("\nDataset is balanced, split, and ready for training!")

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    balance_and_split()
