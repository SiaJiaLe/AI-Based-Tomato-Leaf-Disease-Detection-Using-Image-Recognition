import os

BASE_DIR = os.path.expanduser("~/AI-Based-Tomato-Leaf-Disease-Detection-Using-Image-Recognition/resnet34_model")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
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
