import os
import shutil
import uuid
from pathlib import Path

# The 10 exact folder names for our ResNet34 model
TARGET_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Map YOLO class names to our exact target folders. None = Ignore
CLASS_MAPPING = {
    # real_data_1 classes
    "0": None, # Ignore unknown class 0
    "Early Blight": "Tomato___Early_blight",
    "Early_blight": "Tomato___Early_blight",
    "Healthy": "Tomato___healthy",
    "Iron Deficiency": None, # Ignore
    "Late Blight": "Tomato___Late_blight",
    "Leaf Mold": "Tomato___Leaf_Mold",
    "Leaf_Miner": None, # Ignore
    "Mosaic Virus": "Tomato___Tomato_mosaic_virus",
    "Septoria": "Tomato___Septoria_leaf_spot",
    "Spider Mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": None, # Ignore vague label
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    
    # real_data_2 specific classes (the rest are handled above)
    "Leaf Miner": None, # Ignore
}

# The class names list exactly as they appear in the data.yaml files
DATASET_1_NAMES = ['0', 'Early Blight', 'Early_blight', 'Healthy', 'Iron Deficiency', 'Late Blight', 'Leaf Mold', 'Leaf_Miner', 'Mosaic Virus', 'Septoria', 'Spider Mites', 'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Yellow Leaf Curl Virus']
DATASET_2_NAMES = ['Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold', 'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus']

def process_yolo_dataset(dataset_path, class_names_list, raw_data_dir):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Skipping {dataset_path} - not found.")
        return

    splits = ['train', 'valid', 'test']
    copied_count = 0
    skipped_count = 0

    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'

        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in images_dir.glob('*.*'):
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                skipped_count += 1
                continue
            
            # Read the first bounding box class
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                # Empty label file
                skipped_count += 1
                continue
                
            # Get the class ID of the first bounding box (format: class_id x_center y_center width height)
            try:
                class_id = int(lines[0].split()[0])
                class_name_yaml = class_names_list[class_id]
                target_folder_name = CLASS_MAPPING.get(class_name_yaml)
            except (ValueError, IndexError):
                skipped_count += 1
                continue

            if target_folder_name is None:
                # Class is ignored
                skipped_count += 1
                continue

            # Ensure the target folder exists
            target_dir = Path(raw_data_dir) / target_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy image with a unique name to prevent overwriting
            # Format: real_data_XYZ.jpg
            unique_filename = f"real_{uuid.uuid4().hex[:8]}{image_path.suffix}"
            dest_path = target_dir / unique_filename
            
            shutil.copy2(image_path, dest_path)
            copied_count += 1

    print(f"[{dataset_path.name}] Copied {copied_count} images. Skipped {skipped_count} (ignored classes/no labels).")

if __name__ == "__main__":
    # Base directory is one level up from scripts
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Paths to the downloaded YOLO datasets
    real_data_1_path = base_dir / "real_data_1"
    real_data_2_path = base_dir / "real_data_2"
    
    # Path to the model's raw data directory
    raw_data_dir = base_dir / "resnet34_model" / "data" / "raw"
    
    print("Processing Real World YOLO Datasets...")
    
    process_yolo_dataset(real_data_1_path, DATASET_1_NAMES, raw_data_dir)
    process_yolo_dataset(real_data_2_path, DATASET_2_NAMES, raw_data_dir)
    
    print("\nData merging complete! You can now run the normal prepare_dataset.py script.")
