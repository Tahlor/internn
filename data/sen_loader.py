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
        x = ' '
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


class SentenceDataset(Dataset):
    """Dataset for sentences of 32 characters that can be return as either images or embeddings
        args:
        PATH: path to load the EmnistSampler from
        which: 'Images' or 'Embeddings'
        """

    def __init__(self, PATH=None, which='Images', train=True):
        self.which = which
        if which == 'Images':
            if PATH is not None:
                loaded_data = torch.load(PATH)
                self.images_loaded = loaded_data[0]
                self.sentence_data = loaded_data[1]
            else:
                self.images_loaded = EmnistSampler('sorted_train_emnist.pt')
                self.sentence_data = load_sen_list('text_generation/sent_list.pkl')
                torch.save((self.images_loaded, self.sentence_data), 'sen_img_data.pt')

        elif which == 'Embeddings':
            if PATH is not None:
                loaded_data = torch.load(PATH)
                self.emb_loaded = loaded_data[0]
                self.sentence_data = loaded_data[1]
            else:
                self.emb_loaded = EmbeddingSampler('emb_dataset_train.pt')  # Loads emb's to sample from
                self.sentence_data = load_sen_list('text_generation/sent_list.pkl')  # Loads the sentences
                # torch.save((self.emb_loaded, self.sentence_data), 'sen_emb_data.pt') # Save for quicker loading (Optional)
        self.len = len(self.sentence_data)
        self.space = self.get_space()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Get's a sentences of size 32 or less chars and samples EMNIST or Embeddings for the corresponding character images
        Returns a tensor of an image for every character in the sentence. """

        sentence = self.sentence_data[idx]

        x = list()
        y = list()
        z = list()
        data = None
        sen_len = len(sentence)
        for char in sentence:
            if char == ' ' and self.which == 'Embeddings':
                x.append(self.space[0])
                y.append(self.space[1])
                z.append(self.space[2])
                continue
            char = str.lower(char)
            char_idx = self.letter_to_num(char)
            if self.which == 'Images':
                data = self.images_loaded.sample(char_idx)
            elif self.which == 'Embeddings':
                data = self.emb_loaded.sample(char_idx)
                z.append(data[2])

            x.append(data[0])  # list of tuples of images [0], with labels [1]
            y.append(data[1])

        # Convert into a tensor for each list of tensors and return them in a pair
        x = torch.stack(x)
        if self.which == 'Images': # Embed labels are tensors, Img labels are not.
            y = torch.tensor(y)
        else:
            y = torch.stack(y)

        y = FN.one_hot(y, num_classes=27)
        if self.which == 'Embeddings':  # Emb's pass the output distribution as well
            z = torch.stack(z)
            return x, y, z, sen_len
        return x, y, sen_len

    def createSentenceDataset(self, sen_list_path = None, embedding_sampler_path = None,
                              emnist_sampler_path = None, train = True, output_path=None):
        """
        Function for creating and saving a saved SentenceDataset object. It will store the train and test set of sorted
        emnist images, train and test set of sorted embeddings, and a list of the 32 character sentences from our corpus.
        Args:
            sen_list_path: path to .pkl file that contains the list of 32 char sentences
            embedding_sampler_path: path to .pt file that contains the embeddings
            emnist_sampler_path: path to .pt file that contains the sorted emnist images
            train: boolean value of True (default) or False to determine the train or test set
            output_path: path to where the saved SentenceDataset object will be stored.
        Returns: No return value
        """

        pass

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
        z = torch.tensor([-14.6589, -5.9251, -6.3186, -6.0825, -6.4697, -5.5496, -7.6354,
                          -5.5846, -7.3932, 5.6296, -2.8101, -11.1615, 4.1306, -14.6459,
                          -5.5566, -5.2781, -7.0683, -4.5653, -3.6809, -5.0012, -6.4760,
                          -7.2010, -5.6119, -7.2746, -10.9740, -9.3349, -7.6041],
                         device='cuda:0')
        return x, y, z

    def num_to_letter(self, num):
        x = chr(num + 96)
        if x == '`':
            x = ' '
        return x

    def letter_to_num(self, char):
        if char == ' ':
            return 0
        return ord(char) - 96



class EmbeddingSampler:
    """ For Quick loading and sampling of the embeddings,
        DATA_PATH is the path to the embeddings tuple with [0] embeddings, [1] labels [2] output prob.
        DICT_PATH is the path to the pre-saved dictionary for the EmbeddingSample"""

    def __init__(self, DATA_PATH=None, DICT_PATH=None):
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
                # Stores the embedding(x), label(y), and outputd(z)
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
    def __init__(self, PATH=None, which='train'):
        # Load sorted_emnist if PATH is given
        if PATH:
            self.letters = torch.load(PATH)
            return
        dataset = None
        if which == 'train':
            dataset = torchvision.datasets.EMNIST('./data/emnist', split='letters', train=True, download=True,
                                                  transform=torchvision.transforms.Compose(
                                                      [  # Fix image orientation
                                                          lambda img: F.rotate(img, -90),
                                                          lambda img: F.hflip(img),
                                                          torchvision.transforms.ToTensor(),
                                                          # Convert a PIL image or np.ndarray to tensor
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                          # (mean, std)
                                                      ]))  # , shuffle=True)
        elif which == 'test':
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
            # plot(x[0], x[1])
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

    def sample(self, char_idx):
        if char_idx == 0:
            space_tensor = np.full((1, 28, 28), fill_value=-0.4242)
            return torch.from_numpy(space_tensor), char_idx
        char = num_to_letter(char_idx)
        if char.isupper():
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]) - 1)
        return self.letters[char][img_idx], self.letter_to_num(char)

    def num_to_letter(self, num):
        x = chr(num + 96)
        if x == '`':
            x = ' '
        return x

    def letter_to_num(self, char):
        return ord(char) - 96

def collate_fn(data):
    """
    Args:
        IF Embeddings ->
            data: is a list (len is the batch size) of tuples with (char_embedding, label, length),
            emb: is a tensor of shape([?-32, 512])
            label: is a tensor of shape([?-31, 27]) (one hot encoded)
            length: num of chars in the sentence
        IF Images ->
            data: is a list of tuples with (img, label, length)
            label: is a tensor of shape([?-31, 27]) (one hot encoded)
            length: num of chars in the sentence


    """
    max_len = 32
    num_outputs = 27
    sen_len = data[0][0].size(0)
    num_batches = len(data)
    if len(data[0]) == 3:  # Images
        imgs, labels, lengths = zip(*data)
        new_imgs = []
        for batch_idx in range(num_batches):
            j = imgs[batch_idx].size(0)
            if j == max_len:
                new_imgs.append(imgs[batch_idx])
                continue
            curr_sen_of_imgs = torch.cat((imgs[batch_idx], torch.full((max_len - j, 1, 28, 28), fill_value=-0.4242)))
            new_imgs.append(curr_sen_of_imgs)

        assert(new_imgs[0].shape[0] == 32)

        new_labels = []
        for batch_idx in range(num_batches):
            j = imgs[batch_idx].size(0)
            curr_label = torch.cat((labels[batch_idx], torch.zeros(max_len - j, num_outputs)))
            new_labels.append(curr_label)

        assert(new_labels[0].shape[0] == 32)

        return new_imgs, new_labels, lengths

    else:
        emb, labels, outputs, lengths = zip(*data)  # Unpacks the iterable data object into 3 parts
        emb_len = 512

        # Pad sentences
        new_embs = []
        for batch_idx in range(num_batches):  # Loop through each batch.
            j = emb[batch_idx].size(0)  # num of labels
            curr_emb = torch.cat((emb[batch_idx], torch.zeros(max_len - j, emb_len, device='cuda:0')))
            new_embs.append(curr_emb)
        embs = torch.stack(new_embs)

        new_labels = []
        # Pad each label vectors seperately
        for batch_idx in range(num_batches):
            j = labels[batch_idx].size(0)  # num of labels
            padded_label = torch.cat((labels[batch_idx], torch.zeros(max_len - j, num_outputs, device='cuda:0')))
            new_labels.append(padded_label)
        # Stack label vectors
        labels = torch.stack(new_labels)

        new_outputs = []
        # Pad each output vectors seperately
        for batch_idx in range(num_batches):
            j = outputs[batch_idx].size(0)  # num of labels
            padded_output = torch.cat((outputs[batch_idx], torch.zeros(max_len - j, num_outputs, device='cuda:0')))
            new_outputs.append(padded_output)
        # Stack output vectors
        outputs = torch.stack(new_outputs)

        return embs, labels, outputs, lengths

def sen_embeddings():
    train_dataset = SentenceDataset(PATH='sen_emb_data.pt', which='Embeddings')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        print("Embeddings shape: ", sample[0].shape)
        print("Labels shape: ", sample[1].shape)  # (One hot encoded)
        print("Output shape: ", sample[2].shape)
        print("Sen Lengths: ", sample[3])

        if i_batch == 5:
            exit()

def sen_images():
    train_dataset = SentenceDataset(PATH='sen_img_data.pt', which='Images')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        print("Image shapes: ", sample[0][0].shape)
        print("Labels shape: ", sample[1][0].shape)  # (One hot encoded)
        print("Sen Lengths: ", sample[2][0])

        if i_batch == 5:
            exit()

def example_sen_loader():
    # sen_embeddings()
    sen_images()



# Run Example
# example_sen_loader()
