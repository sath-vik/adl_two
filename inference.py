# inference.py
import torch
import torch.nn.functional as F
import nibabel as nib
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import UNet
from config import IMG_SIZE, VOLUME_SLICES, VOLUME_START_AT, TRAIN_DATASET_PATH, SEGMENT_CLASSES
from data_loading import create_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, n_channels=2, n_classes=4):
    """Load the trained model weights"""
    model = UNet(n_channels, n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # set model to evaluation mode
    return model

def predict_segmentation(model, sample_path):
    """Predict segmentation for a given sample"""
    # Load the NIfTI files
    t1ce_path = sample_path + '_t1ce.nii'
    flair_path = sample_path + '_flair.nii'

    # Extract data
    t1ce = nib.load(t1ce_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()

    # Create empty array
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    # Resize and stack
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(t1ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))

    # Convert data to tensor
    X = torch.tensor(X).permute(0,3,1,2).float().to(device)
    X = X / torch.max(X) # normalize input using torch.max(), which returns a tensor

    with torch.no_grad():
        outputs = model(X)
        probs = F.softmax(outputs, dim=1)
        predicted_mask = torch.argmax(probs, dim=1).cpu().numpy()  # predicted classes [0, 1, 2, 3]

    return predicted_mask


def show_predicted_segmentations(samples_list, slice_to_plot, model):
    """Show predicted segmentation alongside the ground truth"""
    # Choose a random patient
    random_sample = np.random.choice(samples_list)
    # Get path of this patient
    random_sample_path = os.path.join(TRAIN_DATASET_PATH, random_sample, random_sample)

    # Predict patient's segmentation
    predicted_seg = predict_segmentation(model, random_sample_path)

    # Load patient's original segmentation (Ground truth)
    seg_path = random_sample_path + '_seg.nii'
    seg = nib.load(seg_path).get_fdata()

    # Resize original segmentation to the same dimensions of the predictions
    seg=cv2.resize(seg[:,:,slice_to_plot+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)

    # Get predicted classes
    all = predicted_seg[slice_to_plot,:,:] # Predicted categories [0,1,2,3]
    first = (predicted_seg[slice_to_plot,:,:]==1).astype(int)  # Isolation of class 1, Core
    second = (predicted_seg[slice_to_plot,:,:]==2).astype(int)  # Isolation of class 2, Edema
    third = (predicted_seg[slice_to_plot,:,:]==3).astype(int)  # Isolation of class 3, Enhancing

    # Plot Original segmentation & predicted segmentation
    print("Patient number: ", random_sample)
    fig, axstest = plt.subplots(1, 6, figsize=(25, 20))

    # Original segmentation
    axstest[0].imshow(seg)
    axstest[0].set_title('Original Segmentation')

    # All classes
    axstest[1].imshow(all)
    axstest[1].set_title('Predicted Segmentation - all classes')

    # Not tumor class
    axstest[2].imshow((predicted_seg[slice_to_plot,:,:]==0).astype(int))
    axstest[2].set_title('Predicted Segmentation - Not Tumor')

    # Necrotic / Core class
    axstest[3].imshow(first)
    axstest[3].set_title('Predicted Segmentation - Necrotic/Core')

    # Edema class
    axstest[4].imshow(second)
    axstest[4].set_title('Predicted Segmentation - Edema')

    # Enhancing class
    axstest[5].imshow(third)
    axstest[5].set_title('Predicted Segmentation - Enhancing')

    # Add space between subplots
    plt.subplots_adjust(wspace=0.8)
    plt.show()


if __name__ == '__main__':
    # Create dataset for inference
    _, _, test_dataset = create_datasets()
    test_ids = [test_dataset.list_IDs[i] for i in range(0,len(test_dataset.list_IDs))]

    # Load trained model
    model = load_model("best_model.pth") # load from previous training

    # Perform the inference
    show_predicted_segmentations(test_ids, 60, model)
    show_predicted_segmentations(test_ids, 60, model)
    show_predicted_segmentations(test_ids, 65, model)
