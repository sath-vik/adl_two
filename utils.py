# utils.py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

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
    """Collect the training/validation loss/accuracy from TensorBoard logs into a dataframe."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    loss_train = ea.scalars.Items('Loss/train')
    loss_val = ea.scalars.Items('Loss/val')
    learning_rates = ea.scalars.Items('Learning rate')

    epochs = [item.step for item in loss_train]
    loss_train_values = [item.value for item in loss_train]
    loss_val_values = [item.value for item in loss_val]
    learning_rates_values = [item.value for item in learning_rates]

    data = {
        'epochs':epochs,
        'loss_train':loss_train_values,
        'loss_val':loss_val_values,
        'learning_rate': learning_rates_values
    }
    return pd.DataFrame(data)

def calculate_iou_per_class(inputs, masks, num_classes):
    """Computes IoU for each class."""
    ious = []
    for i in range(num_classes):
        mask = (masks == i).float()
        probs = F.softmax(inputs, dim=1)[:,i]
        intersection = (probs * mask).sum()
        union = (probs.sum() + mask.sum()) - intersection
        iou = intersection / (union + 1e-6)  # Added small constant for numerical stability
        ious.append(iou.item())
    return ious

def calculate_miou(inputs, masks, num_classes):
  """Compute Mean IoU (mIoU)."""
  ious = calculate_iou_per_class(inputs, masks, num_classes)
  return np.mean(ious)


def calculate_pixel_accuracy(inputs, masks):
    """Computes Pixel Accuracy."""
    predicted_labels = torch.argmax(inputs, dim=1)
    correct = (predicted_labels == masks).sum().item()
    total = masks.numel()
    return correct / total

def calculate_mean_pixel_accuracy(inputs, masks, num_classes):
  """Computes Mean Pixel Accuracy."""
  total_pixel_accuracy = 0
  for i in range(num_classes):
      current_mask = (masks == i).float()
      total_pixel_accuracy += calculate_pixel_accuracy(inputs[:,i].unsqueeze(1), current_mask.long())
  return total_pixel_accuracy / num_classes