import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob
import openslide 
import json 
import cv2 
import time 
from PIL import Image 
from scipy import ndimage
from PIL import Image, ImageFilter
from PIL import ImageStat 

class TmpPatchesDataset(Dataset):
    """Read all the Patches within the slide"""
    def __init__(self, patch_dir, transform):
        self.patch_dir = patch_dir
        self.transform = transform
        self.patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.patch_files)
    
    @staticmethod
    def parse_patch_name(patch_filename):
        # Remove the file extension
        patch_name = os.path.splitext(patch_filename)[0]
        
        # Split the patch name by underscores
        parts = patch_name.split('_')
        
        # Extract values from the parts list and return as a dictionary
        return {
            'ymin': int(parts[0]),
            'ymax': int(parts[1]),
            'xmin': int(parts[2]),
            'xmax': int(parts[3]),
            'spixel_idx': int(parts[4]),
            'patch_idx': int(parts[5])
        }
     
    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        patch_image = Image.open(patch_path).convert('RGB')  # Open and convert to RGB
        patch_name = os.path.basename(patch_path) 
        
        patch_info = self.parse_patch_name(patch_name)
        if self.transform:
            patch_image = self.transform(patch_image)  # Apply transformations if any

        return {'image': patch_image, 'patch_info': patch_info} 

