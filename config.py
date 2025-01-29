# config.py
import os

# Dataset download URL (we are handling this now in the bash script)
DATASET_URL = None # will skip downloading step in python now
DATASET_ZIP = "brats20-dataset-training-validation.zip" # Name of downloaded zip file
DATASET_DIR = "BraTS2020_TrainingData"  # Directory to extract the data to (relative path)

TRAIN_DATASET_PATH = os.path.join(DATASET_DIR, "MICCAI_BraTS2020_TrainingData")  # Relative path
# Select Slices and Image Size
VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include
IMG_SIZE = 128
BATCH_SIZE = 1 # can be bigger if there is enough memory
EPOCHS = 25

# Define seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3
}
