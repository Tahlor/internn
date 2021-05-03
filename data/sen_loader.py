import os
import pdb
import random
import torch
import pandas
import numpy as np
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

def open_file(path):
    with open(path, "r") as f:
        return "".join(f.readlines())

class SentencesDataset(Dataset):
    """Dataset for sentences of 32 characters"""
    def __init__(self, file):
        self.text = open_file(file) # Stores entire text in data[0]
        self.len = len(self.text)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """Get's random sentences of size 32 chars"""
        a = random.randrange(0, self.len - 32)
        b = a + 32

        return self.text[a:b]

test = SentencesDataset("./text_generation/clean_text.txt")

""" Questions?
Should spaces be included? Spaces are important to make sentences but we don't have images of spaces in emnest do we?

"""