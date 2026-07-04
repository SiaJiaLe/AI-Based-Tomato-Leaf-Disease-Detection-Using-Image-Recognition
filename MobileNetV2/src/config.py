import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SRC_DIR)                 # MobileNetV2/
REPO_ROOT = os.path.dirname(BASE_DIR)                # repo root

DATA_DIR = os.path.join(REPO_ROOT, "resnet34_model", "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Matches resnet34_model/src/config.py's training budget so architecture
# is the only variable in the comparison; augmentation/head/attention/
# loss remain this model's own standard configuration.
STAGE_A_LR = 1e-3
STAGE_A_EPOCHS = 15
STAGE_B_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
