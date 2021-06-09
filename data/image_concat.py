import os, sys
import torch
import random
import torchvision
import torchvision.transforms.functional as F
import torch.nn.functional as FN
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sen_loader import SentenceDataset, num_to_letter, collate_fn



# Load Sen loader - Each data point contains a tensor of images in a sentence

# Loop through and concatonate together.
def combine(imgs, label):
    new_image = trim(imgs[0])
    new_label = num_to_letter(torch.argmax(label[0], dim=0).item())
    for i in range(1, len(imgs)):
        new_image = torch.cat((new_image, imgs[i]), 2)
        new_label += num_to_letter(torch.argmax(label[i], dim=0).item())
        # print(new_image.shape)
    plot(new_image, new_label)


def plot(torchimage, label=None):
    fig = plt.figure()
    npimage = torchimage.permute(1, 2, 0)
    plt.imshow(npimage, cmap='gray', interpolation='none')
    if label is not None:
        plt.title("Ground Truth: {}".format(label))
    plt.show()
    return

def trim(img):
    x_found = None
    for y in range(img.shape[2]):
        for x in range(img.shape[1]):
            if round(img[0][x][y].item(), 4) != -0.4242:
                x_found = x
    pass

def find_leftmost(img):

    pass

def find_rightmost():
    pass

def random_pad(): # Either pad or overlap
    pass

def run():
    train_dataset = SentenceDataset(PATH='sen_img_data.pt', which='Images')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        combine(sample[0][0], sample[1][0])
        # print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        # print("Image shapes: ", sample[0][0].shape)
        # print("Labels shape: ", sample[1][0].shape)  # (One hot encoded)
        # print("Sen Lengths: ", sample[2][0])

        if i_batch == 5:
            exit()

run()


