import torch
import numpy as np
import os
import sys

from torchvision import transforms
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
# target_path = os.path.abspath(os.path.join(current_dir, '../../src/external/wsi_caption/modules'))
target_path = os.path.abspath(os.path.join(current_dir, '../../src/externals/wsi_caption/modules'))
sys.path.append(target_path)
from datasets import TcgaImageDataset

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # Image transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        # Dataset
        if self.dataset_name == 'TCGA':
            self.dataset = TcgaImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        # Init kwargs without distributed samplers
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = images[0].unsqueeze(0)  # Assuming 1 sample per batch
        return images_id, images, torch.LongTensor(reports_ids), torch.FloatTensor(reports_masks)
