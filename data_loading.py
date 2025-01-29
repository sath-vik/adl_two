# data_loading.py
import os
import cv2
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DATASET_PATH, VOLUME_SLICES, VOLUME_START_AT, IMG_SIZE, BATCH_SIZE

class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS data."""

    def __init__(self, list_IDs, transform=None):
        """Initialization."""
        self.list_IDs = list_IDs
        self.transform = transform
        self.scaler = MinMaxScaler()

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.list_IDs) * VOLUME_SLICES

    def __getitem__(self, index):
        """Generates one sample of data."""
        # Determine the case ID and slice index
        case_index = index // VOLUME_SLICES
        slice_index = index % VOLUME_SLICES
        case_id = self.list_IDs[case_index]

        # Load the NIfTI files
        case_path = os.path.join(TRAIN_DATASET_PATH, case_id)
        flair_path = os.path.join(case_path, f'{case_id}_flair.nii')
        t1ce_path = os.path.join(case_path, f'{case_id}_t1ce.nii')
        seg_path = os.path.join(case_path, f'{case_id}_seg.nii')

        flair = nib.load(flair_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # Get the correct slice
        flair_slice = flair[:, :, slice_index + VOLUME_START_AT]
        t1ce_slice = t1ce[:, :, slice_index + VOLUME_START_AT]
        seg_slice = seg[:, :, slice_index + VOLUME_START_AT]

        # Scale and resize
        flair_slice = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        t1ce_slice = cv2.resize(t1ce_slice, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        seg_slice = cv2.resize(seg_slice, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST).astype(np.int64)

        # Normalize the data
        flair_slice = self.scaler.fit_transform(flair_slice.reshape(-1, 1)).reshape(flair_slice.shape)
        t1ce_slice = self.scaler.fit_transform(t1ce_slice.reshape(-1, 1)).reshape(t1ce_slice.shape)


        # Stack the modalities
        image = np.stack([flair_slice, t1ce_slice], axis=-1)

        # Modify mask to have 0, 1, 2, 3
        seg_slice[seg_slice==4] = 3

        # Apply transformations
        if self.transform:
            image = self.transform(image) # transform can be augmentation
            # If transform modifies the mask (rotation) then you will need to do a similar tranformation for it

        # convert to pytorch tensor
        image = torch.tensor(image).permute(2, 0, 1).float() # C, H, W
        seg_slice = torch.tensor(seg_slice).long()  # H, W

        return image, seg_slice


def create_datasets(test_size=0.2, val_size=0.15):
    """Split data into training, validation, and test sets."""
    train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

    def pathListIntoIds(dirList):
      x = []
      for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
      return x

    train_and_test_ids = pathListIntoIds(train_and_val_directories)
    train_and_test_ids = train_and_test_ids[:10] # use only the first 10 patients for testing purposes
    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=test_size)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=val_size)

    train_dataset = BraTSDataset(train_ids)
    val_dataset = BraTSDataset(val_ids)
    test_dataset = BraTSDataset(test_ids)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Create data loaders for each dataset"""

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset and dataloader
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    # Get an item and its shape
    for image, mask in train_loader:
        print("Image batch shape:", image.shape)  # Expected [batch_size, C, H, W]
        print("Mask batch shape:", mask.shape)     # Expected [batch_size, H, W]
        break # Print the shape of first batch, then stop
