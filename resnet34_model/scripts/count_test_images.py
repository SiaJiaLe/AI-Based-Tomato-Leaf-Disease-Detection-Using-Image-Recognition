import os
from pathlib import Path

def count_images():
    base_dir = Path(__file__).resolve().parent.parent
    test_dir = base_dir / "data" / "processed" / "real_environment_test"
    
    if not test_dir.exists():
        print(f"Directory not found: {test_dir}")
        return
        
    print(f"Counting images in: {test_dir.name}\n")
    print("-" * 50)
    
    total_images = 0
    
    # Sort folders alphabetically for clean output
    class_folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    for class_folder in class_folders:
        # Count only image files
        images = [f for f in class_folder.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        count = len(images)
        total_images += count
        
        # Format the output with spacing
        print(f"{class_folder.name:<40} : {count} images")
        
    print("-" * 50)
    print(f"{'TOTAL IMAGES':<40} : {total_images} images")

if __name__ == "__main__":
    count_images()
