import os
import pdb
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data import loaders

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


def open_file(path):
    with open(path, "r") as f:
        return "".join(f.readlines())


def num_to_letter(num):
    return chr(num + 97)


def letter_to_num(char):
    return ord(char) - 96


def plot(data, target):
    pdb.set_trace()
    fig = plt.figure()
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(target))
    return

    """
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    plt.show()
    """


class SentencesDataset(Dataset):
    """Dataset for sentences of 32 characters"""

    def __init__(self, file):
        self.text = open_file(file)  # Stores entire text in data[0]
        self.len = len(self.text)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """Get's random sentences of size 32 chars"""
        a = random.randrange(0, self.len - 32)
        b = a + 32

        return self.text[a:b]


class EmnistSampler:
    def __init__(self):
        # train_loader = torch.utils.data.DataLoader(
        train_dataset = torchvision.datasets.EMNIST('./data/emnist', split='letters', train=True, download=True,
                                                    transform=torchvision.transforms.Compose(
                                                        [  # Composes several transforms together
                                                            torchvision.transforms.RandomPerspective(),
                                                            # %50 percent chance the image perspective will change(distorted)
                                                            torchvision.transforms.RandomRotation(10, fill=(0,)),
                                                            torchvision.transforms.ToTensor(),
                                                            # Convert a PIL image or np.ndarray to tensor
                                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                            # (mean, std)
                                                        ]))  # , shuffle=True)
        letters = {}
        for y_gt, x_tensor in enumerate(train_dataset):
            pdb.set_trace()
            char = num_to_letter(y_gt)
            if char in letters:
                letters[char].append(x_tensor)
            else:
                letters[char] = list()
                letters[char].append(x_tensor)

        self.sample_data = letters
        pdb.set_trace()
        return

    def sample(self):
        return


test = SentencesDataset("./text_generation/clean_text.txt")
emnist_loaded = EmnistSampler()

# train_loader, test_loader = loaders.loader(batch_size_train=100, batch_size_test=1000)

""" Questions?
Should spaces be included? Spaces are important to make sentences but we don't have images of spaces in emnest do we?

"""
