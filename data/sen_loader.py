import os
import pdb
import random
import torch
import torchvision
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

    def __init__(self, file):
        self.emnist_loaded = EmnistSampler()
        # test = self.emnist_loaded.sample('b')
        # plot(test, 2)

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
        for char in sentence:
            #pdb.set_trace()
            x.append((self.emnist_loaded.sample(char), char)) # list of tuples of images [0], with labels [1]
        #pdb.set_trace()
        #for data in x:
        #    print(data[1])
        #pdb.set_trace()
        return x


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
        #pdb.set_trace()
        letters = {}
        for x in train_dataset:
            #pdb.set_trace()

            #plot(x[0], x[1])
            char = num_to_letter(x[1])
            if char in letters:
                letters[char].append(x[0])
            else:
                letters[char] = list()
                letters[char].append(x[0])

        self.letters = letters
        #pdb.set_trace()
        return

    def sample(self, char):
        #pdb.set_trace()
        if char == ' ':
            space_tensor = np.full((1, 28, 28), fill_value=-0.4242)
            return torch.from_numpy(space_tensor)
        if char.isupper():
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]) - 1)
        return self.letters[char][img_idx]

#pdb.set_trace()
sen_dataset = SentencesDataset("./text_generation/clean_text.txt")
train_loader = torch.utils.data.DataLoader(sen_dataset[0][:], batch_size=32, shuffle=False)
#pdb.set_trace()
for i_batch, sample in enumerate(train_loader):
    print("Epoch {}, Batch size {}\n".format(i_batch, 32))
    if i_batch == 0:
        print("Sentence is: {}".format(sample[1][:]))
        for i in range(32):
            plot(sample[0][i], letter_to_num(sample[1][i]))
        pdb.set_trace()

print("Done")


""" TODO

Question for Taylor
- Added my own custom image for spaces
- Format of the batch


"""
