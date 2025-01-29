#!/bin/bash

# Dataset info
DATASET_OWNER="awsaf49" # owner of dataset
DATASET_NAME="brats20-dataset-training-validation" # name of dataset
DATASET_ZIP="brats20-dataset-training-validation.zip"
DATASET_DIR="BraTS2020_TrainingData"

# Check if kaggle is installed
which kaggle > /dev/null
if [[ $? != 0 ]]; then
    echo "kaggle not found. Please install it first: pip install kaggle"
    exit 1
fi

# Check if dataset already exists
if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Downloading dataset using Kaggle API..."

    kaggle datasets download -d "${DATASET_OWNER}/${DATASET_NAME}" -p .

    echo "Extracting dataset..."
    unzip "${DATASET_ZIP}" -d .

    # Remove zip file
    rm -f "${DATASET_ZIP}"

    echo "Dataset downloaded and extracted successfully."
else
    echo "Dataset already downloaded and extracted."
fi

# Run the python script
python main.py
