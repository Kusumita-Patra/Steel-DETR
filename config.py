import os
import torch

# ==============================
# Dataset
# ==============================

# Change this path to your Google Drive dataset location
DATASET_ROOT = "/content/drive/MyDrive/GC10_SSDD"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "cv")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# ==============================
# Classes
# ==============================

CLASS_NAMES = [
    "Crease",
    "Cresent_gap",
    "Inclusion",
    "Oil_spot",
    "Punching_hole",
    "Rolled_pit",
    "Silk_spot",
    "Waist_folding",
    "Water_spot",
    "Welding_line"
]

NUM_CLASSES = len(CLASS_NAMES)

# ==============================
# Image
# ==============================

IMAGE_SIZE = 224

# ==============================
# Training
# ==============================

BATCH_SIZE = 32

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

PIN_MEMORY = True

# ==============================
# Model
# ==============================

BACKBONE = "resnet50"

PRETRAINED = True

BIFPN_CHANNELS = 256

DROPOUT = 0.3

# ==============================
# Device
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Outputs
# ==============================

OUTPUT_DIR = "outputs"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)



# =====================================
# Scheduler
# =====================================

T_MAX = 50

MIN_LR = 1e-6

# =====================================
# Mixed Precision
# =====================================

USE_AMP = True

# =====================================
# Gradient Clipping
# =====================================

GRAD_CLIP = 1.0

# =====================================
# Early Stopping
# =====================================

PATIENCE = 10

# =====================================
# Checkpoints
# =====================================

SAVE_BEST_ONLY = True