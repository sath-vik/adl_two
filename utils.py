# utils.py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

def dice_coef(inputs, masks, smooth = 1.0):
    """Computes the dice coefficient (per class)"""
    class_num = masks.max().item()+1
    total_loss = 0
    for i in range(class_num):
        mask = (masks == i).float()
        probs = F.softmax(inputs, dim=1)[:,i]
        intersection = (probs * mask).sum()
        total_loss +=  (2. * intersection + smooth) / (probs.sum() + mask.sum() + smooth)

    return total_loss / class_num # return avg dice

def dice_loss(inputs, masks, smooth=1.0):
    """Dice loss implementation to train our model"""
    return 1 - dice_coef(inputs, masks, smooth)


def plot_training_history(log_dir):
    """Plot training/validation loss/accuracy from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    loss_train = ea.scalars.Items('Loss/train')
    loss_val = ea.scalars.Items('Loss/val')
    learning_rates = ea.scalars.Items('Learning rate')

    epochs = [item.step for item in loss_train]
    loss_train_values = [item.value for item in loss_train]
    loss_val_values = [item.value for item in loss_val]
    learning_rates_values = [item.value for item in learning_rates]


    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss plot
    axes[0].plot(epochs, loss_train_values, label='Training Loss')
    axes[0].plot(epochs, loss_val_values, label='Validation Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # learning rate plot
    axes[1].plot(epochs, learning_rates_values, label = "Learning Rate")
    axes[1].set_title('Learning rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning rate')

    # learning rate plot
    axes[2].plot(epochs[1:], np.diff(learning_rates_values), label = "Learning Rate")
    axes[2].set_title('Learning rate changes')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning rate')
    axes[2].legend()


    plt.tight_layout()
    plt.show()
