# training.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from data_loading import create_datasets, create_dataloaders
from models import UNet
from utils import dice_coef, dice_loss, plot_training_history
from config import BATCH_SIZE, EPOCHS, IMG_SIZE, SEGMENT_CLASSES
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the loss function
def criterion(outputs, masks):
    """Calculate the loss using categorical crossentropy and dice loss"""

    loss_ce = nn.CrossEntropyLoss()(outputs, masks)
    loss_dice = dice_loss(outputs, masks)
    total_loss = loss_ce + loss_dice
    return total_loss

# Define the training loop
def train_model(model, train_loader, val_loader, optimizer, scheduler, writer):
    """Training loop for the model"""
    model.train()
    best_val_loss = float('inf')

    training_log = [] # collect training log
    for epoch in range(EPOCHS):
        start = time.time()
        epoch_loss = 0
        model.train()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            del images, masks, outputs, loss # free up memory

            if (batch_idx+1) % 5 == 0:
              print(f"Train Batch: {batch_idx+1}/{len(train_loader)} | Loss:{epoch_loss/(batch_idx+1):.4f}")

        # validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_dice_coeff = 0
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice_coeff += dice_coef(outputs, masks).item()

                del images, masks, outputs, loss # free up memory

                if (batch_idx+1) % 5 == 0:
                  print(f"Val Batch: {batch_idx+1}/{len(val_loader)} | Val Loss:{val_loss/(batch_idx+1):.4f}")

        # compute metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_coeff / len(val_loader)
        end = time.time()
        epoch_time = end - start

        training_log.append({ # save this metrics in the training log
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "val_loss": avg_val_loss,
             "val_dice_coef": avg_val_dice,
             "learning_rate": optimizer.param_groups[0]['lr']
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model Saved")

        print(f"Epoch:{epoch + 1}  | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f} seconds")

        # write to tensorboard
        writer.add_scalar("Loss/train", avg_epoch_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(avg_val_loss)

    # Save the log file
    df = pd.DataFrame(training_log)
    df.to_csv("training.log", index = False)


if __name__ == '__main__':
    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    # Initialize the UNet model
    model = UNet(n_channels=2, n_classes=4).to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    # Initialize tensorboard
    writer = SummaryWriter('runs/brats_training')

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, writer)
    writer.close()

    plot_training_history('runs/brats_training')