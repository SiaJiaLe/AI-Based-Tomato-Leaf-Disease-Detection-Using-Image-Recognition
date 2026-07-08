import os
import shutil
from pathlib import Path

def setup_kaggle_dataset():
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Path where you extracted the Kaggle dataset
    kaggle_dir = base_dir / "kaggle_plant_village_dataset"
    
    # Target raw directory
    raw_data_dir = base_dir / "resnet34_model" / "data" / "raw"
    
    print(f"Looking for Kaggle dataset in: {kaggle_dir}")
    
    if not kaggle_dir.exists():
        print("Error: The folder 'kaggle_plant_village_dataset' does not exist.")
        return
        
    # Sometimes datasets extract into a subfolder like 'plantvillage' or 'PlantVillage'
    # We will search for the 10 Tomato folders
    found_folders = False
    
    # Search all subdirectories in the kaggle folder for our target classes
    for root, dirs, files in os.walk(kaggle_dir):
        for dir_name in dirs:
            if dir_name.startswith("Tomato___"):
                found_folders = True
                source_dir = Path(root) / dir_name
                target_dir = raw_data_dir / dir_name
                
                print(f"Copying {dir_name} to raw data folder...")
                
                # Create the target directory if it doesn't exist
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all images inside this class folder
                copied = 0
                for img_path in source_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img_path, target_dir / img_path.name)
                        copied += 1
                        
                print(f"  -> Copied {copied} images.")
                
    if not found_folders:
        print("Error: Could not find any folders starting with 'Tomato___' in the dataset.")
        print("Please make sure you extracted the zip file correctly.")
    else:
        print("\nSuccessfully copied Kaggle dataset to data/raw!")
        print("You can now run 'python resnet34_model/scripts/prepare_dataset.py' to create the train/val/test splits.")

if __name__ == "__main__":
    setup_kaggle_dataset()
