"""
# space should just be a blank image
# dataloader that does embeddings and images
# use indices properly; check collate function


"""

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
RESET=False

# Ignore Warnings
import warnings
from pathlib import Path
from general_tools.utils import get_root

SCRIPT_DIR = (Path(__file__).resolve()).parent
ROOT = get_root("internn")
FOLDER = "data/embedding_datasets/embeddings_v2/"
warnings.filterwarnings("ignore")

import types
def collect(func):
    """ Decorater to return lists instead of generators
    Args:
        func:

    Returns:

    """
    def wrapper(*args, **kwargs):
        r = func(*args, **kwargs)

        if isinstance(r, types.GeneratorType):
            l = list(r)
            if len(l) > 1:
                if isinstance(l[0], str):
                    l = "".join(l)
                return l
            elif l: # if list is non-empty
                return l[0]
        else:
            return r
    return wrapper

def open_file(path):
    with open(path, "r") as f:
        return "".join(f.readlines())

@collect
def num_to_letter(num):
    if isinstance(num, str) and not num.isdigit():
        warnings.warn(f"Expecting a number (not {num})")
        return "?"

    def n2l(n):
        x = chr(n + 96)
        if x == '`':
            x = ' '
        return x
    try:
        for i in num:
            yield n2l(i)
    except:
        yield n2l(num)

@collect
def letter_to_num(char):
    char = char.lower()
    def l2n(char):
        if char == ' ':
            return 0
        return ord(char) - 96

    for i in char:
        yield l2n(i)

def plot(data, label=""):
    fig = plt.figure()
    torchimage = data

    # If channels is first
    if torchimage.shape[0] <=3:
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

def save_dataset(x, y, z, PATH):
    one_hot_labels = FN.one_hot(y)
    torch.save((x, one_hot_labels, z), PATH)


class SentenceDataset(Dataset):
    """Dataset for sentences of 32 characters that can be return as either images or embeddings
        args:
        PATH: path to load the EmnistSampler previously saved
        which: 'Images' or 'Embeddings'
        """

    def __init__(self, PATH=None, which='Images', train=True):
        self.which = which
        self.train = train
        if PATH != None:
            sentence_data, train_images_loaded, test_images_loaded, train_emb_loaded, test_emb_loaded = torch.load(PATH)
            if which == 'Images':
                if train:
                    self.images_loaded = train_images_loaded
                else:
                    self.images_loaded = test_images_loaded
            elif which == 'Embeddings':
                if train:
                    self.emb_loaded = train_emb_loaded
                else:
                    self.emb_loaded = test_emb_loaded
            self.sentence_data = sentence_data
            self.len = len(self.sentence_data)
        #self.space = self.get_space()
        # if which == 'Images':
        #     if PATH is not None:
        #         loaded_data = torch.load(PATH)
        #         self.images_loaded = loaded_data[0]
        #         self.sentence_data = loaded_data[1]
        #     else:
        #         self.images_loaded = EmnistSampler('sorted_train_emnist.pt')
        #         self.sentence_data = load_sen_list('text_generation/sent_list.pkl')
        #         torch.save((self.images_loaded, self.sentence_data), 'sen_img_data.pt')
        #
        # elif which == 'Embeddings':
        #     if PATH is not None:
        #         loaded_data = torch.load(PATH)
        #         self.emb_loaded = loaded_data[0]
        #         self.sentence_data = loaded_data[1]
        #     else:
        #         self.emb_loaded = EmbeddingSampler('emb_dataset_train.pt')  # Loads emb's to sample from
        #         self.sentence_data = load_sen_list('text_generation/sent_list.pkl')  # Loads the sentences
        #         # torch.save((self.emb_loaded, self.sentence_data), 'sen_emb_data.pt') # Save for quicker loading (Optional)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """Get's a sentences of size 32 or less chars and samples EMNIST or Embeddings for the corresponding character images
        Returns a tensor of an image for every character in the sentence.

        Returns:
            images [batch, ch, x, y],
            torch.Size([31, 1, 28, 28]),
            torch.Size([31, 27])
            int

        """

        sentence = self.sentence_data[idx]

        x = list()
        y = list()
        z = list()
        data = None
        sen_len = len(sentence)
        for char in sentence:
            char = str.lower(char)
            char_idx = self.letter_to_num(char)
            if self.which == 'Images':
                data = self.images_loaded.sample(char_idx)
            elif self.which == 'Embeddings':
                data = self.emb_loaded.sample(char_idx)
                z.append(data[2])

            x.append(data[0])  # list of tuples of images [0], with labels [1]
            y.append(char_idx)

        # Convert into a tensor for each list of tensors and return them in a pair
        x = torch.stack(x)


        y = torch.tensor(y)

        y = FN.one_hot(y, num_classes=27)
        if self.which == 'Embeddings':  # Emb's pass the output distribution as well
            z = torch.stack(z)
            return x, y, z, sen_len
        return x, y, sen_len

    def createSentenceDataset(self, sen_list_path,
                              embedding_sampler_path_train,
                              embedding_sampler_path_test,
                              emnist_sampler_path_train,
                              emnist_sampler_path_test,
                              save_path):
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
        ## sentence_data = load_sen_list('text_generation/sent_list.pkl')
        ## images_loaded = EmnistSampler('sorted_train_emnist.pt', 'train')
        ## emb_loaded = EmbeddingSampler('emb_dataset_train.pt', 'train')

        print("Loading sentences...")
        sentence_data = load_sen_list(sen_list_path)
        print("Loading train emnist images...")
        train_images_loaded = EmnistSampler(emnist_sampler_path_train, which='train')
        print("Loading test emnist images...")
        test_images_loaded = EmnistSampler(emnist_sampler_path_test, which='test')
        print("Loading train embeddings...")
        train_emb_loaded = EmbeddingSampler(embedding_sampler_path_train, which='train')
        print("Loading test embeddings...")
        test_emb_loaded = EmbeddingSampler(embedding_sampler_path_test, which='test')
        # SENTENCE_DATA, EmnistSampler, EmnistSampler, EmbeddingSampler, EmbeddingSampler
        torch.save((sentence_data, train_images_loaded, test_images_loaded, train_emb_loaded, test_emb_loaded), save_path)
        print("SAVED AT " + str(save_path))
        return

    @staticmethod
    def letter_to_num(char):
        return letter_to_num(char)

    @staticmethod
    def num_to_letter(num):
        return num_to_letter(num)


class EmbeddingSampler:
    """ For Quick loading and sampling of the embeddings,
        DATA_PATH is the path to the embeddings tuple with [0] embeddings, [1] labels [2] output prob.
        DICT_PATH is the path to the pre-saved dictionary for the EmbeddingSample"""
    def __init__(self, load_path=None, which='train'):
        self.which=which
        if load_path:
            train_letters, test_letters = torch.load(load_path)
            if self.which == 'train':
                self.letters = train_letters
            elif self.which == 'test':
                self.letters = test_letters

    def createEmbeddingSampler(self, train_emb_data_path, test_emb_data_path, save_path= 'train_test_emb_data.pt'):
        """
        Saves the training set and test set of embeddings in an object after they are sorted into dictionaries for easy sampling.
        Pass the path to this object to the EmbeddingSampler constructor.
        Args:
            emb_data_path: path to the object that contains the train/test data sets
            save_path: path where the object will be saved as a .pt file """

        if not Path(save_path).exists() or RESET:
            train_dataset = torch.load(train_emb_data_path)
            embeddings = train_dataset[0]
            labels = train_dataset[1]
            preds_vector = train_dataset[2]
            self.len = len(labels)

            # Store Dictionary of letters {0:[(embedding,letter_idx,preds),
            train_letters = {}
            for i in range(self.len):
                k = labels[i].item()
                # Stores the embedding(x), label(y), and outputd(z)
                if k in train_letters:
                    train_letters[k].append((embeddings[i], k, preds_vector[i]))
                else:
                    train_letters[k] = [(embeddings[i], k, preds_vector[i])]

            test_dataset = torch.load(test_emb_data_path)
            embeddings = test_dataset[0]
            labels = test_dataset[1]
            preds_vector = test_dataset[2]
            self.len = len(labels)

            # Store Dictionary of letters
            test_letters = {}
            for i in range(self.len):
                k = labels[i].item()
                # k = torch.argmax(self.y[i], dim=0).item() one hot encoded
                # Stores the embedding(x), label(y), and outputd(z)
                if k in test_letters:
                    test_letters[k].append((embeddings[i], k, preds_vector[i]))
                else:
                    test_letters[k] = [(embeddings[i], k, preds_vector[i])]

            torch.save((train_letters, test_letters), save_path)
        else:
            train_letters, test_letters = torch.load(save_path)

        if self.which == 'train':
            self.letters = train_letters
        elif self.which == 'test':
            self.letters = test_letters


    # def __init__(self, DATA_PATH=None, DICT_PATH=None, SAVE_PATH = None, which='train'):
    #     if DATA_PATH is None and DICT_PATH is not None:
    #         if which == 'train':
    #             self.letters = torch.load(DICT_PATH)
    #         elif which == 'test':
    #             self.letters = torch.load(DICT_PATH)
    #     else:
    #         dataset = torch.load(DATA_PATH)
    #         self.x = dataset[0]
    #         self.y = dataset[1]
    #         self.z = dataset[2]
    #         self.len = len(self.y)
    #
    #         # Store Dictionary of letters
    #         letters = {}
    #         for i in range(self.len):
    #             k = self.y[i].item()
    #             # Stores the embedding(x), label(y), and outputd(z)
    #             if k in letters:
    #                 letters[k].append((self.x[i], self.y[i], self.z[i]))
    #             else:
    #                 letters[k] = [(self.x[i], self.y[i], self.z[i])]
    #
    #         self.letters = letters
    #         if SAVE_PATH is None:
    #             SAVE_PATH = 'saved_ ' + which + '_emb_dict.pt'
    #         torch.save(self.letters, SAVE_PATH)


    def sample(self, char):
        """ Returns a random embedding from our dictionary for the specified letter"""
        # if char.isupper():
        #     char = char.lower()
        emb_idx = random.randrange(0, len(self.letters[char]))
        return self.letters[char][emb_idx]

class EmnistSampler:
    def __init__(self, PATH=None, which='train', save_path=None):
        # Load sorted_emnist if PATH is given
        if PATH:
            self.letters = torch.load(PATH)
            return
        dataset = None
        if which == 'train':
            dataset = torchvision.datasets.EMNIST(ROOT / 'data/emnist', split='letters', train=True, download=True,
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
            dataset = torchvision.datasets.EMNIST(ROOT / 'data/emnist', split='letters', train=False, download=True,
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

        # Add a space to the image EMNIST dataset
        mm = [(torch.zeros(x[0].shape) + torch.min(x[0]))]
        letters[' '] = mm

        self.letters = letters
        # Save dictionary of lists of char images
        if which == 'train':
            torch.save(self.letters, save_path)
        else:
            torch.save(self.letters, save_path)
        return

    def sample(self, char_idx):
        # Space goes here
        # if char_idx == 0:
        #     space_tensor = np.full((1, 28, 28), fill_value=-0.4242)
        #     return torch.from_numpy(space_tensor), char_idx
        char = num_to_letter(char_idx)
        if char.isupper():
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]))
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

# EXAMPLE CREATING data object to load in SentenceDataset
def create_datasets(save_folder,
                    load_emb_train,
                    load_emb_test,
                    ):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    ## Creating Emnist Sampler (Dictionary) from random emnist images
    print("Loading test and train emnist images into dictionary for sampling...")

    # Save EMNIST samplers if they don't exist already
    train_path = save_folder / 'train_sorted_emnist.pt'
    if not train_path.exists() or RESET:
        train_emnist = EmnistSampler(which='train', PATH=None, save_path=train_path)
        test_emnist = EmnistSampler(which='test', PATH=None, save_path=save_folder / 'test_sorted_emnist.pt')

    ## Creating Embeddings Sampler (Dictionary) from precalculated embeddings
    ## Calculated Embeddings in main_both.py in calc_embeddings()
    print("Loading test and train embeddings into dictionary for sampling...")
    # With load_path EmbeddingSampler will load both train and test and leave
    train_emb = EmbeddingSampler(which='train', load_path=None) #'train_test_emb_dict.pt'
    train_emb.createEmbeddingSampler(train_emb_data_path=load_emb_train,
                                     test_emb_data_path=load_emb_test,
                                     save_path=save_folder / 'train_test_emb_dict.pt')
    ## To get test
    ## test_emb = EmbeddingSampler(which='test', load_path='train_test_emb_dict.pt')

    ## Everything previous is essential to have the necessary files to load
    print("Loading Sentence Dataset...")
    obj = SentenceDataset()
    obj.createSentenceDataset(sen_list_path=ROOT / 'data/text_generation/sent_list.pkl',
                              emnist_sampler_path_train= save_folder / 'train_sorted_emnist.pt',
                              emnist_sampler_path_test=save_folder / 'test_sorted_emnist.pt',
                              embedding_sampler_path_train= save_folder / 'train_test_emb_dict.pt',
                              embedding_sampler_path_test=save_folder / 'train_test_emb_dict.pt',
                              save_path= save_folder / "train_test_sentenceDataset.pt")
    print("Finished!")


# EXAMPLE
def sen_embeddings():
    train_dataset = SentenceDataset(PATH=SCRIPT_DIR / 'sen_emb_data.pt', which='Embeddings')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        print("Epoch {}, Batch size {}\n".format(i_batch, 2))
        print("Embeddings shape: ", sample[0].shape)
        print("Labels shape: ", sample[1].shape)  # (One hot encoded)
        print("Output shape: ", sample[2].shape)
        print("Sen Lengths: ", sample[3])

        if i_batch == 5:
            exit()

# EXAMPLE
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
    create_datasets(ROOT / FOLDER,
                    ROOT / (FOLDER + "train_emb_dataset.pt"),
                    ROOT / (FOLDER + "test_emb_dataset.pt")
                    )
    # sen_images()
    # sen_embeddings()

def load_sen_loader():
    sd = SentenceDataset(ROOT / (FOLDER + 'train_test_sentenceDataset.pt'))
    return sd

# Run Example
if __name__ == '__main__':
    #example_sen_loader()
    #import importlib, sen_loader
    #importlib.reload(sen_loader); from sen_loader import *
    RESET=True
    example_sen_loader()
    sd = load_sen_loader()
    m = next(iter(sd))

"""
import importlib, sen_loader
sd = sen_loader.SentenceDataset('train_test_sentenceDataset.pt')
importlib.reload(sen_loader); from sen_loader import *
sd = load_sen_loader()
m = next(iter(sd))
num_to_letter(torch.argmax(m[-2], axis=1))
"""
