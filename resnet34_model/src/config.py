import os

BASE_DIR = os.path.expanduser("~/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition/resnet34_model")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
# Held-out real-world field photos, sibling to processed/ — never used
# for training/val, only for the separate real-world generalization
# evaluation (see real_world_generalization_plan.md, Pillar 5).
REAL_WORLD_DIR = os.path.join(BASE_DIR, "data", "real_environment_test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMAGE_SIZE = 224
BATCH_SIZE = 32
STAGE_A_LR = 1e-3
STAGE_A_EPOCHS = 15
STAGE_B_LR = 1e-4
STAGE_B_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
