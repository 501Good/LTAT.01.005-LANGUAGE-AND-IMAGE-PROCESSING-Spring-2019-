import torch
from torch.utils.data import Dataset
import os
from skimage import io
import csv
import json
import numpy as np


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, csv_file, split, vocab, transform=None, file_format='torch'):
        """
        :param data_folder: folder where data files are stored
        :param csv_file: filename of the csv 
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.file_format = file_format
        assert self.file_format in {'torch', 'numpy', 'image'}

        self.data_folder = data_folder

        self.data = json.load(open(csv_file, encoding='utf-8'))

        self.vocab = vocab

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.data)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        if self.file_format == 'torch':
            img_name = os.path.join(self.data_folder, self.data[i][0] + '.pt')
            img = torch.load(img_name)
        elif self.file_format == 'numpy':
            img_name = os.path.join(self.data_folder, self.data[i][0] + '.npy')
            img = np.load(img_name)
        else:
            img_name = os.path.join(self.data_folder, self.data[i][0] + '.jpg')
            img = io.imread(img_name)
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print(img.shape, i, img_name)
                raise IndexError

        caption = torch.LongTensor(self.data[i][2])
        caplen = torch.LongTensor([self.data[i][3]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = caption
            return img, caption, caplen, all_captions, img_name

    def __len__(self):
        return self.dataset_size
