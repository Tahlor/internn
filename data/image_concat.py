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
class SentenceDatasetAttached(Dataset):
    def __init__(self, path=None):
        pass
    def __len__(self):
        pass
    def __getitem__(self, item):
        pass
    def saveNewDataset(self):
        pass


# Loop through and concatonate together.
def combine(imgs, label):
    new_image = trim(imgs[0])
    new_label = num_to_letter(torch.argmax(label[0], dim=0).item())
    #plot(new_image, new_label)
    for i in range(1, len(imgs)):
        if i == 32:
            print('here')
        test = torch.argmax(label[i], dim=0).item()
        curr_label = num_to_letter(test)
        pad = random.randrange(-2, 5)
        #pad = -2
        if curr_label == ' ':
            new_image = torch.cat((new_image, torch.full((1, 28, pad + 5), fill_value=-0.4242)), dim=2)
            new_label += curr_label
            continue
        if pad == 0:
            new_image = torch.cat((new_image, trim(imgs[i])), dim=2)
        elif pad < 0:
            # pad represents the number of columns that overlap
            # store cols that will overlap
            overlap = torch.full((28, abs(pad)), fill_value=-0.4242)
            left_overlap = new_image[0][:, pad:]
            right_overlap = trim(imgs[i])[0][:, :abs(pad)]
            for j in range(28):
                for k in range(abs(pad)):
                    if left_overlap[j][k] >= right_overlap[j][k]:
                        overlap[j][k] = left_overlap[j][k]
                    else:
                        overlap[j][k] = right_overlap[j][k]
            index = new_image.shape[2] + pad
            new_image = torch.cat((new_image, trim(imgs[i])[:,:, abs(pad):]), dim=2)
            new_image[0][:, index:index+abs(pad)] = overlap
        else: # pad > 0
            new_image = torch.cat((new_image, torch.full((1, 28, pad), fill_value=-0.4242)), dim=2)
            new_image = torch.cat((new_image, trim(imgs[i])), 2)
        new_label += curr_label
    #plot(new_image, new_label)
    return new_image, new_label

def random_pad(low, high):
    return random.randrange(low, high)

def plot(torchimage, label=None):
    fig = plt.figure()
    npimage = torchimage.permute(1, 2, 0)
    plt.imshow(npimage, cmap='gray', interpolation='none')
    if label is not None:
        plt.title("Ground Truth: {}".format(label))
    plt.show()
    return

def trim(img):
    left_edge = find_leftmost(img)
    right_edge = find_rightmost(img)
    new_img = img[:, :, left_edge+1:right_edge]
    #plot(new_img, 't')
    return new_img

def find_leftmost(img):
    x_start = 27
    for y in range(img.shape[1]):
        for x in range(img.shape[2]):
            if round(img[0][y][x].item(), 4) != -0.4242:
                if x < x_start:
                    x_start = x
    return x_start

def find_rightmost(img):
    x_end = 0
    for y in range(img.shape[2]):
        for x in range(img.shape[1]):
            if round(img[0][y][x].item(), 4) != -0.4242:
                if x > x_end:
                    x_end = x
    return x_end

def run():
    train_dataset = SentenceDataset(PATH='sen_img_data.pt', which='Images')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, collate_fn=collate_fn, shuffle=False)
    max = 0
    for i_batch, sample in enumerate(train_loader):
        # print("Batch Number: ", i_batch)
        image, label = combine(sample[0][0], sample[1][0])
        #plot(image, label)
        length = image.shape[2]
        if length > max:
            max = length

        # print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        # print("Image shapes: ", sample[0][0].shape)
        # print("Labels shape: ", sample[1][0].shape)  # (One hot encoded)
        # print("Sen Lengths: ", sample[2][0])

        if i_batch % 50 == 0:
            print("Batch Number: ", i_batch)
            print("Max --->  ", max)

            #exit()

run()


