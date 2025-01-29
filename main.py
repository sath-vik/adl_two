# main.py
import torch
import os
import requests
import zipfile
from training import train_model
from data_loading import create_datasets, create_dataloaders
from models import UNet
from utils import plot_training_history
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate_model
from inference import show_predicted_segmentations, load_model
from config import EPOCHS, DATASET_URL, DATASET_ZIP, DATASET_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_and_extract():
    """Downloads and extracts the dataset."""
    global DATASET_URL # Indicate we are using global variable

    if not os.path.exists(DATASET_DIR):
       if DATASET_URL is not None:
          print("Downloading dataset...")
          response = requests.get(DATASET_URL, stream=True)
          if "drive.google.com" in DATASET_URL:
            response.raise_for_status()
            for key, value in response.headers.items():
                if key.lower() == 'location':
                    DATASET_URL=value
                    response = requests.get(DATASET_URL, stream=True)
                    break
          with open(DATASET_ZIP, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
               f.write(chunk)
       else:
           print("Skipping download step. Using local zip file.") # added local zip file message

       print("Extracting dataset...")
       try:
         with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
           zip_ref.extractall(DATASET_DIR)
       except zipfile.BadZipFile:
         print(f"Error: downloaded file {DATASET_ZIP} is not a valid zip file. Please check the file.")
         return
       if DATASET_URL is not None:
          os.remove(DATASET_ZIP)
       print("Dataset downloaded and extracted successfully.")
    else:
        print("Dataset already downloaded and extracted.")

if __name__ == '__main__':
    # 0. Download the data
    download_and_extract()

    # 1. Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    # 2. Initialize the UNet model
    model = UNet(n_channels=2, n_classes=4).to(device)

    # 3. Define the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    # 4. Initialize tensorboard
    writer = SummaryWriter('runs/brats_training')

    # 5. Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, writer)
    writer.close()

    # 6. Plot training history
    plot_training_history('runs/brats_training')

    # 7. Evaluate the model
    model_eval = load_model("best_model.pth") # load best trained model
    test_dice = evaluate_model(model_eval, test_loader)
    print(f"Test Dice coefficient: {test_dice:.4f}")

    # 8. Visualize predictions
    test_ids = [test_dataset.list_IDs[i] for i in range(0,len(test_dataset.list_IDs))]
    show_predicted_segmentations(test_ids, 60, model_eval) # use the loaded model
    show_predicted_segmentations(test_ids, 60, model_eval)
    show_predicted_segmentations(test_ids, 65, model_eval)
