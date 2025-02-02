# evaluation.py
import torch
import numpy as np
import torch.nn.functional as F
from data_loading import create_datasets, create_dataloaders
from models import UNet
from utils import dice_coef, calculate_iou_per_class, calculate_miou, calculate_pixel_accuracy, calculate_mean_pixel_accuracy
from config import SEGMENT_CLASSES, IMG_SIZE, VOLUME_START_AT, TRAIN_DATASET_PATH, VOLUME_SLICES
from inference import load_model
import cv2
import matplotlib.pyplot as plt
import os
import nibabel as nib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader, num_classes):
    """Evaluate the model's performance on the test set and calculate metrics"""
    model.eval()  # set to eval mode
    total_dice = 0
    total_miou = 0
    total_pixel_accuracy = 0
    total_mean_pixel_accuracy = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            total_dice += dice_coef(outputs, masks).item()
            total_miou += calculate_miou(outputs, masks, num_classes)
            total_pixel_accuracy += calculate_pixel_accuracy(outputs, masks)
            total_mean_pixel_accuracy += calculate_mean_pixel_accuracy(outputs, masks, num_classes)



    avg_dice = total_dice / len(test_loader)
    avg_miou = total_miou / len(test_loader)
    avg_pixel_accuracy = total_pixel_accuracy / len(test_loader)
    avg_mean_pixel_accuracy = total_mean_pixel_accuracy / len(test_loader)
    return avg_dice, avg_miou, avg_pixel_accuracy, avg_mean_pixel_accuracy

def predictByPath(model, case_path, case):
    """Get prediction of all slices"""
    from config import VOLUME_SLICES, IMG_SIZE, VOLUME_START_AT
    files = next(os.walk(case_path))[2]
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii');
    flair=nib.load(vol_path).get_fdata()

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii');
    ce=nib.load(vol_path).get_fdata()


    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE,IMG_SIZE))

    X = torch.tensor(X).permute(0,3,1,2).float().to(device)
    X = X / torch.max(X)
    with torch.no_grad():
      outputs = model(X)
      probs = F.softmax(outputs, dim=1)
      predicted_mask = torch.argmax(probs, dim=1).cpu().numpy()  # predicted classes [0, 1, 2, 3]

    return predicted_mask

def showPredictsById(model, case, start_slice = 60):
    """Show slices with ground truth and predicted mask"""
    from config import TRAIN_DATASET_PATH, IMG_SIZE, VOLUME_START_AT
    path = f"{TRAIN_DATASET_PATH}/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(model, path,case)

    core = (p[:,:,:]==1).astype(int)
    edema= (p[:,:,:]==2).astype(int)
    enhancing = (p[:,:,:]==3).astype(int)

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1,7, figsize = (18, 50))

    for i in range(7): # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')

    axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt=cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3) # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')

    axarr[2].imshow(p[start_slice,:,:], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes predicted')

    axarr[3].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    # Display segmentation
    axarr[6].imshow(np.argmax(p[start_slice,:,:], axis=-1), cmap = "Reds", interpolation='none', alpha=0.3)
    axarr[6].title.set_text(f'all classes predicted (argmax)')
    plt.show()



def classEvaluation(model, case, eval_class = 1, slice_at = 40):
    """Per class evaluation of the prediction with the ground truth"""
    from config import TRAIN_DATASET_PATH, IMG_SIZE, VOLUME_START_AT
    path = f"{TRAIN_DATASET_PATH}/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    p = predictByPath(model, path,case)


    gt[gt != eval_class] = 0 # make it so there is only class 0 and the eval_class
    resized_gt = cv2.resize(gt[:,:,slice_at+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    plt.figure()
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(resized_gt, cmap="gray")
    axarr[0].title.set_text('ground truth')
    axarr[1].imshow(p[slice_at,:,:]==eval_class, cmap="gray")
    axarr[1].title.set_text(f'predicted class: {SEGMENT_CLASSES[eval_class]}')
    plt.show()


if __name__ == '__main__':
    # Create the datasets and dataloaders
    _, _, test_dataset = create_datasets()
    test_loader = create_dataloaders(None, None, test_dataset)[2] # get only test dataloader
    test_ids = [test_dataset.list_IDs[i] for i in range(0,len(test_dataset.list_IDs))] # get test_ids

    # Load the trained model
    model = load_model("best_model.pth")

    # Evaluate the model
    test_dice, test_miou, test_pixel_accuracy, test_mean_pixel_accuracy = evaluate_model(model, test_loader, num_classes=4)

    print(f"Test Dice coefficient: {test_dice:.4f}")
    print(f"Test Mean IoU: {test_miou:.4f}")
    print(f"Test Pixel Accuracy: {test_pixel_accuracy:.4f}")
    print(f"Test Mean Pixel Accuracy: {test_mean_pixel_accuracy:.4f}")

    # Visual evaluation by slice
    showPredictsById(model, case=test_ids[0][-3:])
    showPredictsById(model, case=test_ids[1][-3:])
    showPredictsById(model, case=test_ids[2][-3:])
    showPredictsById(model, case=test_ids[3][-3:])

    # Per class evaluation
    classEvaluation(model, case=test_ids[3][-3:], eval_class=1, slice_at=40)
    classEvaluation(model, case=test_ids[3][-3:], eval_class=2, slice_at=40)
    classEvaluation(model, case=test_ids[3][-3:], eval_class=3, slice_at=40)