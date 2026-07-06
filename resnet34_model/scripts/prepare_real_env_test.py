import os
import shutil
import uuid
from pathlib import Path

# Dataset 6: Extract like normal (ignore 'Tomato leaf')
MAPPING_6 = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": None,
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
}

# Dataset 7: Extract Target Spot only
MAPPING_7 = {
    "Bacterial Spot": None,
    "Early_Blight": None,
    "Healthy": None,
    "Late_blight": None,
    "Leaf Mold": None,
    "Target_Spot": "Tomato___Target_Spot",
    "black spot": None,
}

# Dataset 8: Extract Target Spot and Spider Mites only (Note: they aren't in your yaml names, but we map them just in case)
MAPPING_8 = {
    "Early Blight": None,
    "Healthy": None,
    "Late Blight": None,
    "Leaf Miner": None,
    "Leaf Mold": None,
    "Mosaic Virus": None,
    "Septoria": None,
    "Yellow Leaf Curl Virus": None,
    "Spider Mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Target_Spot": "Tomato___Target_Spot",
}

# Dataset 9: Extract like normal (ignore Leaf Miner)
MAPPING_9 = {
    "Early Blight": "Tomato___Early_blight",
    "Healthy": "Tomato___healthy",
    "Late Blight": "Tomato___Late_blight",
    "Leaf Miner": None,
    "Leaf Mold": "Tomato___Leaf_Mold",
    "Mosaic Virus": "Tomato___Tomato_mosaic_virus",
    "Septoria": "Tomato___Septoria_leaf_spot",
    "Spider Mites": "Tomato___Spider_mites Two-spotted_spider_mite",
    "Yellow Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

# Target Spot Dataset
MAPPING_TS = {
    "TS": "Tomato___Target_Spot"
}

# Class names exactly as they appear in the data.yaml files
NAMES_6 = ['Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf']
NAMES_7 = ['Bacterial Spot', 'Early_Blight', 'Healthy', 'Late_blight', 'Leaf Mold', 'Target_Spot', 'black spot']
NAMES_8 = ['Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold', 'Mosaic Virus', 'Septoria', 'Yellow Leaf Curl Virus']
NAMES_9 = ['Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold', 'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus']
NAMES_TS = ['TS']

def process_yolo_dataset(dataset_path, class_names_list, class_mapping, target_base_dir):
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
                skipped_count += 1
                continue
                
            # Get the class ID of the first bounding box
            try:
                class_id = int(lines[0].split()[0])
                class_name_yaml = class_names_list[class_id]
                target_folder_name = class_mapping.get(class_name_yaml)
            except (ValueError, IndexError):
                skipped_count += 1
                continue

            if target_folder_name is None:
                # Class is explicitly ignored by our mapping
                skipped_count += 1
                continue

            # Ensure the target folder exists inside real_environment_test
            target_dir = Path(target_base_dir) / target_folder_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Move or Copy image with a unique name
            unique_filename = f"real_test_{uuid.uuid4().hex[:8]}{image_path.suffix}"
            dest_path = target_dir / unique_filename
            
            # We copy instead of move here just in case you need to re-run
            shutil.copy2(image_path, dest_path)
            copied_count += 1

    print(f"[{dataset_path.name}] Extracted {copied_count} images. Skipped {skipped_count} (ignored classes/no labels).")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Paths to the downloaded YOLO datasets
    ds6 = base_dir / "real_dataset_6"
    ds7 = base_dir / "real_dataset_7"
    ds8 = base_dir / "real_dataset_8"
    ds9 = base_dir / "real_dataset_9"
    ds_ts = base_dir / "real_dataset_target_spot"
    
    # Target directory for the real environment test set
    target_dir = base_dir / "resnet34_model" / "data" / "processed" / "real_environment_test"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting Real Environment Test Dataset (Target Spot ONLY)...")
    print(f"Target folder: {target_dir}")
    
    # process_yolo_dataset(ds6, NAMES_6, MAPPING_6, target_dir)
    # process_yolo_dataset(ds7, NAMES_7, MAPPING_7, target_dir)
    # process_yolo_dataset(ds8, NAMES_8, MAPPING_8, target_dir)
    # process_yolo_dataset(ds9, NAMES_9, MAPPING_9, target_dir)
    
    # ONLY process the TS dataset
    process_yolo_dataset(ds_ts, NAMES_TS, MAPPING_TS, target_dir)
    
    print("\nExtraction complete! Target spot test images have been placed into real_environment_test.")
