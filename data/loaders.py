# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Letters).
This code trains both NNs as two different models.
This code randomly changes the learning rate to get a good result.
@author: Dipu
"""


import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
from sen_loader import *
import numpy as np


def advanced_loader():
    pass

def sentence_loader():
    train_dataset = SentencesDataset(None, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)

    test_dataset = SentencesDataset(None, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


def loader(batch_size_train = 100,
           batch_size_test = 1000,
           *args,
           **kwargs):

    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.EMNIST('./data/emnist', split='letters', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.RandomPerspective(),
                                   torchvision.transforms.RandomRotation(10, fill=(0,)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.EMNIST('./data/emnist', split='letters', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    print(example_data.shape)
    plot(example_data, example_targets)
    return train_loader, test_loader

def plot(example_data, example_targets):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for i in range(6):
      plt.subplot(2,3,i+1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Ground Truth: {}".format(example_targets[i]))
      plt.xticks([])
      plt.yticks([])
    plt.savefig("./output/")
    fig
