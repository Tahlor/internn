# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Letters).
This code trains both NNs as two different models.
This code randomly changes the learning rate to get a good result.
@author: Dipu
"""
import pdb

import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


def advanced_loader():
    pass

def loader(batch_size_train = 100,
           batch_size_test = 1000,
           *args,
           **kwargs):

    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.EMNIST('./data/emnist', split='letters', train=True, download=True,
                                 transform=torchvision.transforms.Compose([ # Composes several transforms together
                                   torchvision.transforms.RandomPerspective(), # %50 percent chance the image perspective will change(distorted)
                                   torchvision.transforms.RandomRotation(10, fill=(0,)),
                                   torchvision.transforms.ToTensor(), #Convert a PIL image or np.ndarray to tenosry
                                   torchvision.transforms.Normalize( # (mean, std)
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

    pdb.set_trace()
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
    plt.show()
    #plt.savefig("./output/")
    #fig
