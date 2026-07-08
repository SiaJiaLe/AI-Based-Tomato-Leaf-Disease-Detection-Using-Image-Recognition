import os
import random
import shutil
from pathlib import Path

def inject_real_world_data():
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"
    
    test_dir = processed_dir / "real_environment_test"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    
    if not test_dir.exists():
        print(f"Error: {test_dir} does not exist!")
        return
        
    print("Injecting 50% of real-world test images into the training pipeline...\n")
    
    total_moved_to_train = 0
    total_moved_to_val = 0
    total_kept_in_test = 0
    
    class_folders = [d for d in test_dir.iterdir() if d.is_dir()]
    
    for class_folder in class_folders:
        images = [f for f in class_folder.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not images:
            continue
            
        # Shuffle images randomly
        random.shuffle(images)
        
        # Calculate splits: 50% stays in test. 
        # The remaining 50% is split: 80% to train, 20% to val.
        total_imgs = len(images)
        half_idx = total_imgs // 2
        
        kept_test_imgs = images[:half_idx]
        imgs_to_move = images[half_idx:]
        
        train_split_idx = int(len(imgs_to_move) * 0.8)
        
        train_imgs = imgs_to_move[:train_split_idx]
        val_imgs = imgs_to_move[train_split_idx:]
        
        # Ensure target directories exist
        (train_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
        
        # Move to Train
        for img in train_imgs:
            dest = train_dir / class_folder.name / img.name
            shutil.move(str(img), str(dest))
            total_moved_to_train += 1
            
        # Move to Val
        for img in val_imgs:
            dest = val_dir / class_folder.name / img.name
            shutil.move(str(img), str(dest))
            total_moved_to_val += 1
            
        total_kept_in_test += len(kept_test_imgs)
        
        print(f"[{class_folder.name}] Kept {len(kept_test_imgs)} | Moved to Train: {len(train_imgs)} | Moved to Val: {len(val_imgs)}")
        
    print("-" * 50)
    print(f"Total Kept in Test Set : {total_kept_in_test}")
    print(f"Total Injected to Train: {total_moved_to_train}")
    print(f"Total Injected to Val  : {total_moved_to_val}")
    print("\nInjection complete! The model will now see real-world textures during training.")

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    inject_real_world_data()
