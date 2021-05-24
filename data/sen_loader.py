import os
import pdb
import random
import torch
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import pandas
import string
import numpy as np
import matplotlib.pyplot as plt
from torch import float32
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
    return chr(num + 96)


def letter_to_num(char):
    return ord(char) - 96


def plot(data, label):
    fig = plt.figure()
    torchimage = data
    npimage = torchimage.permute(1, 2, 0)
    plt.imshow(npimage, cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(num_to_letter(label)))
    plt.show()
    return


class SentencesDataset(Dataset):
    """Dataset for sentences of 32 characters"""

    def __init__(self, PATH, which):
        self.emnist_loaded = EmnistSampler(PATH, which)
        # test = self.emnist_loaded.sample('b')
        # plot(test, 2)
        file = "./text_generation/clean_text.txt"
        self.text = open_file(file)
        self.len = len(self.text)
        random.seed = random.random()


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Get's random sentences of size 32 chars and samples EMNIST for the corresponding character images
        Returns a list of an image for every character in the sentence. """
        a = random.randrange(0, self.len - 32)
        b = a + 32
        sentence = self.text[a:b]
        x = list()
        y = list()
        for char in sentence:
            x.append(self.emnist_loaded.sample(char))  # list of tuples of images [0], with labels [1]
            y.append(letter_to_num(char))

        # Convert into a tensor for each list of tensors and return them in a pair
        x = torch.stack(x)
        y = torch.from_numpy(np.array(y))
        return x, y


class EmnistSampler:
    def __init__(self, PATH = None, which='train'):
        # Load sorted_emnist if PATH is given
        if PATH:
            self.letters = torch.load(PATH)
            return

        if which == 'train':
            dataset = torchvision.datasets.EMNIST('./data/emnist', split='letters', train=True, download=True,
                                                        transform=torchvision.transforms.Compose(
                                                            [   # Fix image orientation
                                                                lambda img: F.rotate(img, -90),
                                                                lambda img: F.hflip(img),
                                                                torchvision.transforms.ToTensor(),
                                                                # Convert a PIL image or np.ndarray to tensor
                                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                                # (mean, std)
                                                            ]))  # , shuffle=True)
        else:
            dataset = torchvision.datasets.EMNIST('./data/emnist', split='letters', train=False, download=True,
                                                  transform=torchvision.transforms.Compose(
                                                      [  # Fix image orientation
                                                          lambda img: F.rotate(img, -90),
                                                          lambda img: F.hflip(img),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                      ]))  # , shuffle=True)

        # Store Training Dataset
        letters = {}
        for x in dataset:
            #plot(x[0], x[1])
            char = num_to_letter(x[1])
            if char in letters:
                letters[char].append(x[0])
            else:
                letters[char] = list()
                letters[char].append(x[0])
        self.letters = letters

        # Save dictionary of lists of char images
        if which == 'train':
            torch.save(self.letters, 'sorted_train_emnist.pt')
        else:
            torch.save(self.letters, 'sorted_test_emnist.pt')

        return

    def sample(self, char):
        if char == ' ':
            space_tensor = np.full((1, 28, 28), fill_value=-0.4242)
            return torch.from_numpy(space_tensor)
        if char.isupper():
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]) - 1)
        return self.letters[char][img_idx]


def example_sen_loader():
    train_dataset = SentencesDataset('./sorted_train_emnist.pt', 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)

    test_dataset = SentencesDataset('./sorted_test_emnist.pt', 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        print("Epoch {}, Batch size {}\n".format(i_batch, 32))
        if i_batch == 0:
            for i in range(32):
                test = sample[0][0][i]
                plot(test, sample[1][0][i])
            break
        exit()

# Run Example
example_sen_loader()





