#!/usr/bin/env python
# coding: utf-8

# # Custom Segmentation Dataset Class for aerial images 

# Prerequisites
import os
from pathlib import Path
import cv2

from torch.utils.data import Dataset # for creating custom dataset


# ### Custom Dataset
class SegmentationDataset(Dataset):

    def __init__(self, path_name) -> None:
        super().__init__()
        self.image_names = os.listdir(f"{path_name}/images")  # list of all image names
        self.image_paths = [f"{path_name}/images/{i}" for i in self.image_names]  # list of all image paths
        self.mask_names = os.listdir(f"{path_name}/masks")
        self.mask_paths = [f"{path_name}/masks/{i}" for i in self.mask_names]

        # Remove items that do not exist in boths images and masks folders
        self.image_stem = [Path(i).stem for i in self.image_paths]   # get file names without extension
        self.mask_stem = [Path(i).stem for i in self.mask_paths]

        # Take union to get both files  
        self.image_mask_stem = set(self.image_stem) & set(self.mask_stem)

        self.image_paths = [i for i in self.image_paths if (Path(i).stem in self.image_mask_stem)] # filter image paths
        self.mask_paths = [i for i in self.mask_paths if (Path(i).stem in self.image_mask_stem)] # 

    def __len__(self):
        # Return the number of image-mask pairs
        return len(self.image_mask_stem)
    
    def convert_mask(self, mask):
        # Convert original mask to 0-5 classes
        mask[mask==155] = 0 # unlabeled
        mask[mask == 44] = 1 # building
        mask[mask == 91] = 2 # land
        mask[mask == 171] = 3 # water
        mask[mask == 172] = 4 # road
        mask[mask == 212] = 5 # vegetation
        return mask           

    def __getitem__(self, index):
        # Return image & mask
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        image = image.transpose(2, 0, 1)  # Make it BS, C, H, W
        mask = cv2.imread(self.mask_paths[index], 0)   # read as Grayscale
        mask = self.convert_mask(mask)
        return image, mask
        

