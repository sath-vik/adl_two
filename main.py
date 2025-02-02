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
from evaluation import evaluate_model, showPredictsById, classEvaluation
from inference import show_predicted_segmentations, load_model
from config import EPOCHS, DATASET_URL, DATASET_ZIP, DATASET_DIR, SEGMENT_CLASSES
import matplotlib.pyplot as plt
import numpy as np

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

def plot_all_metrics(df):
  """Plot training loss, validation loss, and learning rate over time."""
  epochs = df["epochs"]
  loss_train = df["loss_train"]
  loss_val = df["loss_val"]
  learning_rate = df["learning_rate"]

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Loss plot
  axes[0].plot(epochs, loss_train, label='Training Loss')
  axes[0].plot(epochs, loss_val, label='Validation Loss')
  axes[0].set_title('Loss over Epochs')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Loss')
  axes[0].legend()

  # learning rate plot
  axes[1].plot(epochs, learning_rate, label = "Learning Rate")
  axes[1].set_title('Learning rate')
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Learning rate')

    # learning rate plot
  axes[2].plot(epochs[1:], np.diff(learning_rate), label = "Learning Rate")
  axes[2].set_title('Learning rate changes')
  axes[2].set_xlabel('Epoch')
  axes[2].set_ylabel('Learning rate')
  axes[2].legend()
  plt.show()



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
    df = plot_training_history('runs/brats_training')
    plot_all_metrics(df) # new plot code with matplotlib

    # 7. Evaluate the model
    model_eval = load_model("best_model.pth") # load best trained model
    test_dice, test_miou, test_pixel_accuracy, test_mean_pixel_accuracy = evaluate_model(model_eval, test_loader, num_classes=4)
    print(f"Test Dice coefficient: {test_dice:.4f}")
    print(f"Test Mean IoU: {test_miou:.4f}")
    print(f"Test Pixel Accuracy: {test_pixel_accuracy:.4f}")
    print(f"Test Mean Pixel Accuracy: {test_mean_pixel_accuracy:.4f}")


    # 8. Visualize predictions
    test_ids = [test_dataset.list_IDs[i] for i in range(0,len(test_dataset.list_IDs))]
    showPredictsById(model_eval, case=test_ids[0][-3:])
    showPredictsById(model_eval, case=test_ids[1][-3:])
    showPredictsById(model_eval, case=test_ids[2][-3:])
    showPredictsById(model_eval, case=test_ids[3][-3:])

    # Per class evaluation
    classEvaluation(model_eval, case=test_ids[3][-3:], eval_class=1, slice_at=40)
    classEvaluation(model_eval, case=test_ids[3][-3:], eval_class=2, slice_at=40)
    classEvaluation(model_eval, case=test_ids[3][-3:], eval_class=3, slice_at=40)