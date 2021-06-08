import os
import pdb
import random
import torch
import torchvision
import torchvision.transforms.functional as F
import torch.nn.functional as FN
import matplotlib.pyplot as plt
import pandas
import string
import numpy as np
import matplotlib.pyplot as plt
from torch import float32
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data import loaders
import pickle

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")

def open_file(path):
    with open(path, "r") as f:
        return "".join(f.readlines())

def num_to_letter(num):
    x = chr(num + 96)
    if x == '`':
        x = "padding"
    return x

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

def load_sen_list(PATH):
    open_file = open(PATH, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list



class SentenceEmbDataset(Dataset):
    """Dataset for sentences of 32 characters"""

    def __init__(self, PATH=None, which=None):
        #self.emnist_loaded = EmnistSampler(PATH, which) # Loads imgs to sample from
        # test = self.emnist_loaded.sample('b')
        # plot(test, 2)
        if PATH is not None:
            loaded_data = torch.load('sen_emb_data.pt')
            self.emb_loaded = loaded_data[0]
            self.sentence_data = loaded_data[1]
        else:
            self.emb_loaded = EmbeddingSampler('emb_dataset_train.pt') # Loads emb's to sample from
            self.sentence_data = load_sen_list('text_generation/sent_list.pkl') # Loads the sentences
        self.len = len(self.sentence_data)
        self.space = self.get_space()
        random.seed = random.random()
        torch.save((self.emb_loaded, self.sentence_data), 'sen_emb_data.pt')



    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Get's a sentences of size 32 or less chars and samples EMNIST or Embeddings for the corresponding character images
        Returns a tensor of an image for every character in the sentence. """
        # Gets a random string of chars of size 32 (OLD WAY)
        # a = random.randrange(0, self.len - 32)
        # b = a + 32
        # sentence = self.text[a:b]

        sentence = self.sentence_data[idx]

        x = list()
        y = list()
        z = list()
        sen_len = len(sentence)
        for char in sentence:
            if char == ' ':
                x.append(self.space[0])
                y.append(self.space[1])
                z.append(self.space[2])
                continue
            char = str.lower(char)
            char_idx = letter_to_num(char)
            data = self.emb_loaded.sample(char_idx)
            x.append(data[0])  # list of tuples of images [0], with labels [1]
            y.append(data[1])
            z.append(data[2])

        # Convert into a tensor for each list of tensors and return them in a pair
        x = torch.stack(x)
        y = torch.stack(y)
        y = FN.one_hot(y, num_classes=27)
        z = torch.stack(z)
        return x, y, z, sen_len

    def get_space(self):
        x = torch.tensor([0.0000, 0.0000, 0.3574, 0.5313, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.4303, 0.0000, 0.2127, 0.0000, 0.0000,
         0.0000, 0.1401, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0214, 0.1339,
         0.0000, 0.0000, 0.0000, 0.0000, 0.3682, 0.0000, 0.0000, 0.0000, 0.0491,
         0.0000, 0.5024, 0.0000, 0.3816, 0.0000, 0.0000, 0.0000, 0.0000, 0.0056,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1895, 0.0000, 0.0000,
         0.0000, 0.2752, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.1586, 0.0000, 0.0000, 0.0000, 0.1493, 0.0000, 0.0000, 0.0000,
         0.0797, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0501,
         0.3497, 0.0000, 0.0000, 0.0000, 0.0000, 0.2252, 0.0000, 0.1844, 0.0000,
         0.0000, 0.0000, 0.0741, 0.0153, 0.3897, 0.0000, 0.0000, 0.0000, 0.2780,
         0.0000, 0.0000, 0.0000, 0.2395, 0.0065, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1791, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4308,
         0.0000, 0.0000, 0.0364, 0.0000, 0.0000, 0.0000, 0.0000, 0.0576, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2687, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0725, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2193, 0.0000, 0.0000, 0.0000, 0.2794, 0.0000, 0.1826,
         0.0498, 0.0000, 0.1186, 0.0000, 0.1226, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2473, 0.3170,
         0.0000, 0.0000, 0.0000, 0.0563, 0.3153, 0.0000, 0.0000, 0.0000, 0.0000,
         0.3826, 0.0000, 0.0000, 0.1507, 0.0000, 0.0000, 0.0000, 0.1992, 0.0000,
         0.2189, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0234, 0.2287, 0.0000, 0.0000, 0.1799, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0601, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2003, 0.0000, 0.0000, 0.0000, 0.4333, 0.0000, 0.0000,
         0.0000, 0.0000, 0.2006, 0.0000, 0.0000, 0.0017, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0569, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.3776, 0.0000, 0.0000, 0.0446, 0.0000, 0.0998, 0.2578,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.2529, 0.0000, 0.0094, 0.0000, 0.0000, 0.0000, 0.0000, 0.1833,
         0.0000, 0.1734, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.1456, 0.2167, 0.2203, 0.0000, 0.0000, 0.0431, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.4517, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.6141, 0.0025, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0875, 0.1866, 0.0000, 0.0000, 0.0000, 0.4353, 0.0000, 0.0000,
         0.1974, 0.0000, 0.2797, 0.0000, 0.0000, 0.0000, 0.0000, 0.3020, 0.0529,
         0.0000, 0.3837, 0.1463, 0.2151, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.1675, 0.0000, 0.0000, 0.0000, 0.2391, 0.0000, 0.3017, 0.0000, 0.0000,
         0.0000, 0.0000, 0.4239, 0.0000, 0.2377, 0.0000, 0.0000, 0.0000, 0.3258,
         0.0000, 0.1397, 0.0000, 0.1069, 0.0000, 0.0000, 0.0000, 0.0000, 0.1659,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0632, 0.0000,
         0.3668, 0.0747, 0.1799, 0.0000, 0.3867, 0.0000, 0.0000, 0.0000, 0.3599,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0387, 0.0000, 0.0043, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3010, 0.0000,
         0.3191, 0.0351, 0.0000, 0.0000, 0.5424, 0.0269, 0.0000, 0.0000, 0.0000,
         0.3978, 0.1781, 0.0000, 0.0000, 0.0284, 0.0000, 0.0921, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.1443, 0.0000, 0.0000, 0.1536, 0.0000,
         0.0332, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2565, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3464, 0.0000,
         0.0000, 0.0951, 0.0000, 0.0771, 0.0000, 0.0000, 0.0000, 0.0000],
       device='cuda:0')
        y = torch.tensor(0, device='cuda:0')
        z = torch.tensor([-14.6589,  -5.9251,  -6.3186,  -6.0825,  -6.4697,  -5.5496,  -7.6354,
          -5.5846,  -7.3932,   5.6296,  -2.8101, -11.1615,   4.1306, -14.6459,
          -5.5566,  -5.2781,  -7.0683,  -4.5653,  -3.6809,  -5.0012,  -6.4760,
          -7.2010,  -5.6119,  -7.2746, -10.9740,  -9.3349,  -7.6041],
       device='cuda:0')
        return x, y, z


class EmbeddingSampler:
    """ For Quick loading and sampling of the embeddings,
        DATA_PATH is the path to the embeddings tuple with [0] embeddings, [1] labels [2] output prob.
        DICT_PATH is the path to the pre-saved dictionary for the EmbeddingSample"""
    def __init__(self, DATA_PATH = None, DICT_PATH = None):
        if DATA_PATH is None and DICT_PATH is not None:
            self.letters = torch.load(DICT_PATH)
        else:
            dataset = torch.load(DATA_PATH)
            self.x = dataset[0]
            self.y = dataset[1]
            self.z = dataset[2]
            self.len = len(self.y)

            # Store Dictionary of letters
            letters = {}
            for i in range(self.len):
                k = self.y[i].item()
                if k in letters:
                    letters[k].append((self.x[i], self.y[i], self.z[i]))
                else:
                    letters[k] = [(self.x[i], self.y[i], self.z[i])]

            self.letters = letters
            if DICT_PATH is None:
                DICT_PATH = 'saved_emb_dict.pt'
            torch.save(self.letters, DICT_PATH)

    """ Returns a random embedding from our dictionary for the specified letter"""
    def sample(self, char):
        # if char.isupper():
        #     char = char.lower()
        emb_idx = random.randrange(0, len(self.letters[char]) - 1)
        return self.letters[char][emb_idx]

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
        # if which == 'train':
        #     torch.save(self.letters, 'sorted_train_emnist.pt')
        # else:
        #     torch.save(self.letters, 'sorted_test_emnist.pt')

        return

    def sample(self, char):
        if char == ' ':
            space_tensor = np.full((1, 28, 28), fill_value=-0.4242)
            return torch.from_numpy(space_tensor)
        if char.isupper():
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]) - 1)
        return self.letters[char][img_idx]

def collate_fn(data):
    """
    Args:
        data: is a list (len is the batch size) of tuples with (char_embedding, label, length),

        emb: is a tensor of shape([?-32, 512])
        label: is a tensor of shape([?-31, 27]) (one hot encoded)

    """

    emb, labels, outputs, lengths = zip(*data) # Unpacks the iterable data object into 3 parts
    max_len = 32
    num_outputs = 27
    emb_len = 512
    sen_len = data[0][0].size(0)
    num_batches = len(data)

    new_labels = []
    # Pad each label vectors seperately
    for batch_idx in range(num_batches):
        padded_label = torch.zeros(max_len, num_outputs).int()
        j = labels[batch_idx].size(0) # num of labels
        padded_label = torch.cat([labels[batch_idx], torch.zeros(max_len - j, num_outputs, device='cuda:0')])
        new_labels.append(padded_label)
    # Stack label vectors
    labels = torch.stack(new_labels)

    new_outputs = []
    # Pad each output vectors seperately
    for batch_idx in range(num_batches):
        padded_output= torch.zeros(max_len, num_outputs).int()
        j = outputs[batch_idx].size(0)  # num of labels
        padded_output = torch.cat([outputs[batch_idx], torch.zeros(max_len - j, num_outputs, device='cuda:0')])
        new_outputs.append(padded_output)
    # Stack output vectors
    outputs = torch.stack(new_outputs)

    # Pad sentences
    new_embs = []
    for batch_idx in range(num_batches):  # Loop through each batch.
        curr_emb = torch.zeros(max_len, emb_len).int()  # batch = 1 ([32, 512)], batch = 2 ([64, 512])
        j = emb[batch_idx].size(0)  # num of labels
        curr_emb = torch.cat([emb[batch_idx], torch.zeros(max_len - j, emb_len, device='cuda:0')])
        new_embs.append(curr_emb)
    embs = torch.stack(new_embs)

    return embs, labels, outputs, lengths

def example_sen_loader():

    train_dataset = SentenceEmbDataset()#'sen_emb_data.pt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        print("Embeddings shape: ", sample[0].shape)
        print("Labels shape: ", sample[1].shape)  # (One hot encoded)
        print("Output shape: ", sample[2].shape)
        print("Sen Lengths: ", sample[3])

        if i_batch == 5:
            exit()

# Run Example
example_sen_loader()




