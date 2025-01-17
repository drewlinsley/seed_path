import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.datasets import ImageFolder


class Classification(Dataset):
    """Dataset for image classification tasks."""

    def __init__(self, data_path, mu, std, transforms=None, return_path=False):
        """
        Initialize the SoothsayerDataset.

        Args:
            data_path (str): Path to the data directory.
            transforms (callable, optional): Optional transform to be applied on a sample.
            split (str): One of 'train', 'test', 'human', or 'val'.
            task (str): One of 'perspective' or 'depth'.
            return_path (bool): Whether to return the image path with each item.
        """        
        pos = os.path.join(data_path, "pos")
        neg = os.path.join(data_path, "neg")
        pos_files = glob(os.path.join(pos, "*.png"))
        neg_files = glob(os.path.join(neg, "*.png"))
        self.files = np.asarray(pos_files + neg_files)
        self.labels = np.concatenate([np.ones(len(pos_files)), np.zeros(len(neg_files))])
        self.transforms = transforms
        self.return_path = return_path
        self.mu = mu
        self.std = std

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get item at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) or (image, label, img_path) if return_path is True.
        """
        img_path = self.files[idx]
        image = np.asarray(Image.open(img_path)).astype(np.float32)[None]
        label = int(self.labels[idx])
        image = (image - self.mu) / self.std
        image = torch.from_numpy(image.astype(np.float32))
        if self.transforms:
            image = self.transforms(image)

        if self.return_path:
            return image, label, img_path
        return image, label
