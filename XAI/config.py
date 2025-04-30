"""
Configuration settings for the skin lesion classification project.
"""

import os
from pathlib import Path

# Project directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, 
                 EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset paths
HAM10000_METADATA = RAW_DATA_DIR / "HAM10000_metadata.csv"
HAM10000_IMAGES_PART1 = RAW_DATA_DIR / "HAM10000_images_part_1"
HAM10000_IMAGES_PART2 = RAW_DATA_DIR / "HAM10000_images_part_2"

# Model parameters
MODEL_INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
LR_MIN = 0.00001

# Class names and mappings
CLASS_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

# melanoma 4 , basal cell carcinoma 1 , and squamous cell carcinoma

NUM_CLASSES = len(CLASS_NAMES)

# Training related
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)  # Train, Validation, Test split ratios
RANDOM_SEED = 42
