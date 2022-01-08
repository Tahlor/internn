"""
# space should just be a blank image
# dataloader that does embeddings and images
# use indices properly; check collate function
"""
PRECALCULATE_EMBEDDING = True
import re
import datasets
import copy
import os
import pdb
import random
import torch
from torch import tensor
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

"""
Creates:
train_test_sentenceDataset.pt
train_sorted_emnist.pt
test_sorted_emnist.pt
"""

# Ignore Warnings
import warnings
from pathlib import Path
from general_tools.utils import get_root

SCRIPT_DIR = (Path(__file__).resolve()).parent
ROOT = get_root("internn")
FOLDER = "data/embedding_datasets/embeddings_V5/"
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


def filter_with_punctuation():
    lower_case = re.compile(r'[^a-z0-9 \.]+')
    double_space = re.compile(r'\s\s+')
    space_period = re.compile(r'\s+\.')
    return lambda sentence: space_period.sub(".",
                                            double_space.sub(" ",
                                            lower_case.sub("", sentence)).strip())

def filter_only_lowercase(include_digits=True):
    if include_digits:
        lower_case = re.compile(r'[^a-z0-9 ]+')
    else:
        lower_case = re.compile(r'[^a-z ]+')

    double_space = re.compile(r'\s\s+')
    return lambda sentence: double_space.sub(" ",lower_case.sub("", sentence)).strip()


FILTERS = {"lowercase": lambda: filter_only_lowercase(include_digits=False),
           "lowercase_with_digits": lambda: filter_only_lowercase(include_digits=True),
           "filter_with_punctuation": filter_with_punctuation,
           }

def default_normalize(input, dim=None):
    return input

NORMALIZE_FUNC = {
    "L2": torch.nn.functional.normalize,
    "softmax": torch.nn.functional.softmax,
    "default": default_normalize # lambda x, *_: x
}

class SentenceDataset(Dataset):

    """Dataset for sentences of 32 characters that can be return as either images or embeddings
        args:
        PATH: path to load the SentenceDataset previously saved
        which: 'Images' or 'Embeddings'
        active_mode:
                    full sequence - assume the entire sequence is being predicted
                    single character - mask / predict only one character
                    multicharacter

        TODO: Add multicharacter masking to train faster
        TODO: Add collate function
        """

    def __init__(self,
                 PATH,
                 which='Images',
                 train=True,
                 train_mode="full sequence",
                 sentence_length=32,
                 sentence_filter="lowercase",
                 vocab_size=27,
                 normalize="L2",
                 multicharacter_number=5,
                 alphabet="abcdefghi jklmnopqrstuvwxyz"):
        """

        Args:
            PATH:
            which:
            train:
            train_mode:
            sentence_length:
            sentence_filter:
            vocab_size:
            normalize (str): default, L2, softmax
        which = "Both"  is not implemented

        """
        print(f"Sentence Dataset Path: {PATH}")
        self.which = which
        self.train = train
        self.train_mode = train_mode
        self.parse_train_mode(self.train_mode)
        self.multicharacter_number = multicharacter_number
        self.sentence_length = sentence_length
        self.filter_sentence = FILTERS[sentence_filter]()
        self.vocab_size = vocab_size
        self.normalize_func = NORMALIZE_FUNC[normalize]
        self.alphabet = alphabet
        assert which in ["Images", "Embeddings", "Both"]

        self.sentence_data, self.train_images_loaded, self.test_images_loaded, self.train_emb_loaded, self.test_emb_loaded = torch.load(PATH)
        self.sentence_data_train, self.sentence_data_test = datasets.load_dataset('bookcorpus', split=['train[:85%]', 'train[85%:]'])
        self.load_images()
        if self.train:
            self.set_train_mode()

        self.N = self._embd_mean = self._embd_std = 0

        ## ALWAYS USE TRAINING MASK EMBEDDINGS
        self.embd_mean, self.embd_std = self.train_emb_loaded.embd_mean, self.train_emb_loaded.embd_std

        self.b, self.bb = 0.9,0.99

    def save_small(self, path):

        for x in self.train_images_loaded, self.test_images_loaded, self.train_emb_loaded, self.test_emb_loaded:
            for key in x.letters:
                x.letters[key] = x.letters[key][:10]
        m = [self.sentence_data[:1000], self.train_images_loaded, self.test_images_loaded, self.train_emb_loaded, self.test_emb_loaded]
        torch.save(m, path)

    def parse_train_mode(self, train_mode=None):
        if train_mode is None:
            train_mode = self.train_mode
        for tm in ["full sequence", "single character", "multicharacter"]:
            if tm in train_mode:
                self.active_mode = tm
                break

        # Train mode must be assigned
        assert self.active_mode
        self.train_mode_maskchar_option = {}
        if not train_mode:
            self.train_mode_maskchar_option[""] = 1
            self.train_mode_maskchar_list = ["",1]
        else:
            total_num = 0
            for opt in ["MASK_CHAR", "DONT_ATTEND_TO_MASK_CHAR", "RANDOM_CHAR", "USE_CORRECT_CHAR", "MEAN_EMBEDDING"]:
                if opt in train_mode:
                    num = re.search(f"({opt})([\._0-9]*)", train_mode)
                    if num and num[2].isdigit():
                        num = int(num[2])
                    elif opt in ["MASK_CHAR", "RANDOM_CHAR", "MEAN_EMBEDDING"]: # no number provided, default is 50% random mask character
                        num = 50
                    else:
                        num = 0
                    # Add the variants with weight
                    if num:
                        self.train_mode_maskchar_option[opt] = num
                        total_num += num
            #self.train_mode_maskchar_option = {k: v / total_num for k, v in self.train_mode_maskchar_option.items()}
            self.train_mode_maskchar_list = [(k,v/total_num) for k,v in self.train_mode_maskchar_option.items()]
        if False:
            print(f"Dataloader mode is {self.active_mode, self.train_mode_maskchar_option}")

    def load_images(self, train=True, which=""):
        if not which:
            which = self.which

        # if which in ['Images', "Both"]:
        #     if train:
        #         self.images_loaded = self.train_images_loaded
        #     else:
        #         self.images_loaded = self.test_images_loaded
        #     self.sample_image = self.images_loaded.sample(0)
        #
        # if which in ['Embeddings', "Both"]:
        if train:
            self.emb_loaded = self.train_emb_loaded
        else:
            self.emb_loaded = self.test_emb_loaded
        self.sample_embedding = self.emb_loaded.sample(0)

    def __len__(self):
        return self.len

    def get_sentence(self, sentence):
        # Always begins and ends with a space - WHY???
        filtered_sentence = " " + self.filter_sentence(sentence["text"]) + " "

        try:
            start = random.randint(0, len(filtered_sentence) - self.sentence_length)
            new_start = filtered_sentence[start:].find(" ")+start+1
            fs = filtered_sentence[new_start:new_start+self.sentence_length]
            new_end = fs.rfind(" ")
            fs = fs[:new_end]
        except Exception as e:
            fs = filtered_sentence[:self.sentence_length]
            new_end = fs.rfind(" ") if fs.rfind(" ") > 0 else None
            fs = fs[:new_end].strip()

        return fs

    def set_train_mode(self):
        self.sentence_data = self.sentence_data_train
        self.load_images(train=True)
        self.len = len(self.sentence_data)
        self.parse_train_mode(self.train_mode)

    def set_test_mode(self):
        self.sentence_data = self.sentence_data_test
        self.load_images(train=False)
        self.len = len(self.sentence_data)
        self.parse_train_mode("full sequence")

    def sentence_to_data(self, sentence):
        embeddings = list()
        images = list()
        gt_idxs = list()
        vgg_logit = list()

        for char in sentence:
            char = str.lower(char)
            char_idx = self.letter_to_num(char)
            data = self.emb_loaded.sample(char_idx) # 512-embedding, char-idx, one-hot-27 vector
            vgg_logit.append(data[2])
            images.append(data[3])
            embeddings.append(data[0])  # list of tuples of images [0], with labels [1]
            gt_idxs.append(char_idx)
        return embeddings, images, gt_idxs, vgg_logit

    def get_random_char(self, char=None):
        if char is None:
            char = random.choice(self.alphabet)
        char_idx = self.letter_to_num(char)
        data = self.emb_loaded.sample(char_idx)
        return data # 0 = embedding, 2 = logits, 3 = image

    # This is stupid
    def update_exponential_average(self, embeddings):
        # self.embd_mean = self.embd_std = self.N = 0
        embd_std, embd_mean = torch.std_mean(embeddings, 0)
        self.N += 1
        denom, denom2 = 1 - self.b ** self.N, 1 - self.bb ** self.N
        self._embd_mean = (self._embd_mean * self.b + (1 - self.b) * embd_mean)
        self._embd_std = (self._embd_std * self.bb + (1 - self.bb) * embd_std)

        self.embd_mean = self._embd_mean / denom
        self.embd_std = self._embd_std / denom2

    def __getitem__(self, idx, mask_idx=-1):
        """ Get's a sentences of size 32 or less chars and samples EMNIST or Embeddings for the corresponding character images
        Returns a tensor of an image for every character in the sentence.

        Returns:
            images [batch, ch, x, y], 
            torch.Size([31, 1, 28, 28]),
            torch.Size([31, 27])
            int

        """
        while True:
            sentence = self.get_sentence(self.sentence_data[idx])
            if len(sentence) >= 4:
                break
            else:
                idx = random.randint(0,self.len)

        # for i in range(0, 1000):
        #     sentence = self.get_sentence(self.sentence_data_train[i])
        #     print(sentence)

        data, embedding, image = None, None, None
        sen_len = len(sentence)
        vgg_text = ""

        # Sample images from sentence letters
        embeddings, images, gt_idxs, vgg_logit = self.sentence_to_data(sentence)

        # Convert into a tensor for each list of tensors and return them in a pair
        if len(images):
            images = torch.stack(images)
        if len(embeddings):
            embeddings = torch.stack(embeddings)

        if self.which in ('Images', 'Both'): # Embed labels are tensors, Img labels are not.
            gt_idxs = torch.tensor(gt_idxs)
        else:
            if gt_idxs and isinstance(gt_idxs[0], int):
                gt_idxs = torch.tensor(gt_idxs)
            else:
                gt_idxs = torch.stack(gt_idxs)

        gt_one_hot = FN.one_hot(gt_idxs, num_classes=self.vocab_size)

        # Provide attention and label masks
        attention_mask = torch.ones([sen_len])
        if self.active_mode.endswith("character"):
            choices = min(self.multicharacter_number, int(sen_len / 6)) if self.active_mode == "multicharacter" else 1
            mask_idx = mask_char_idx = np.random.choice(sen_len, choices, replace=False)
            old_embeddings = embeddings
            # Compute exponential average of dataset
            if not PRECALCULATE_EMBEDDING and ("MASK_CHAR" or "MEAN_EMBEDDING" in self.train_mode_maskchar_option.keys()):
                self.update_exponential_average(embeddings)

                # Sample some mask_idx
                #mask_char_count = torch.sum(torch.rand(len(mask_idx)) > prob)
            if self.train_mode_maskchar_list[0][0]:
                mask_char_counts = np.random.multinomial(len(mask_idx),[p[1] for p in self.train_mode_maskchar_list])
                embeddings = embeddings.clone()
                start = 0
                for i, (version, prob) in enumerate(self.train_mode_maskchar_list):
                    mask_char_count = mask_char_counts[i]
                    end = start + mask_char_count
                    if version in ["MASK_CHAR","RANDOM_CHAR", "MEAN_EMBEDDING"]:
                        mask_char_idx = mask_idx[start:end] # the indices that have been replaced with a mask_char
                        start = end
                        if mask_char_count:
                            if version=="MASK_CHAR":
                                mask_chars = (torch.randn(
                                    [mask_char_count, *self.embd_mean.shape]) + self.embd_mean) + self.embd_std
                            elif version=="MEAN_EMBEDDING":
                                mask_chars = self.embd_mean
                            elif version=="RANDOM_CHAR":
                                mask_chars = torch.cat([self.get_random_char()[0] for i in range(mask_char_count)])

                            embeddings[mask_char_idx] = mask_chars

            attention_mask[mask_idx] = 0
            masked_gt = torch.zeros([sen_len],dtype=torch.long) - 100
            masked_gt[mask_idx] = gt_idxs[mask_idx]  # everything else to not predict is -100
        elif self.active_mode == "full sequence":
            masked_gt = gt_idxs
            mask_char_idx = [] # the indices that have been replaced with a mask_char -- NONE!
            mask_idx = [x for x in range(0,sen_len)] # the indices that are being predicted
        else:
            raise Exception(f"Unknown active_mode {self.active_mode}")
        masked_gt = masked_gt.type(torch.LongTensor)

        if self.which in ('Embeddings', 'Both'):  # Emb's pass the output distribution as well
            vgg_logit = torch.stack(vgg_logit)
            vgg_text = get_text(vgg_logit)
            embeddings = self.normalize_func(embeddings, dim=-1)
            vgg_logit  = self.normalize_func(vgg_logit, dim=-1)

        num_preds = masked_gt[masked_gt != -100].shape[0]

        return {"data":embeddings if self.which in ('Embeddings', 'Both') else images,
                "embedding":embeddings,
                "image": images,
                "gt_one_hot":gt_one_hot,
                "length":sen_len,
                "attention_mask": attention_mask, # [1,1,1,0,0,]
                "masked_gt": masked_gt, # full sequence: [5,2,0,3,-100,-100...]; multichar: [-100,-100,0,-100,-100,-100...]
                "text": sentence,
                "vgg_logits": vgg_logit,
                "vgg_text": vgg_text,
                "gt_idxs": gt_idxs, # GTs [1,2,3,17,...]
                "mask_idx": mask_idx, # the indices that have been masked, i.e., not predicted, can't be attended to, [2,5,6]
                "mask_char_idx": mask_char_idx,
                "num_preds": num_preds} # the number of predictions being made

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
        self.load_path=load_path
        if load_path:
            d = torch.load(load_path)
            self.setup_from_dict(d)

    def createEmbeddingSampler(self, train_emb_data_path, test_emb_data_path, save_path= 'train_test_emb_data.pt'):
        """
        Saves the training set and test set of embeddings in an object after they are sorted into dictionaries for easy sampling.
        Pass the path to this object to the EmbeddingSampler constructor.
        Args:
            emb_data_path: path to the object that contains the train/test data sets
            save_path: path where the object will be saved as a .pt file """

        if not Path(save_path).exists() or RESET:
            train_test_letters = []
            train_test_master_index = {}
            for ii, var in enumerate(["train", "test"]):
                dataset = test_emb_data_path if var=="test" else train_emb_data_path
                train_dataset = torch.load(dataset)
                embeddings = train_dataset[0]
                labels = train_dataset[1]
                preds_vector = train_dataset[2]
                images = train_dataset[3]
                self.len = len(labels)

                self.train_letters = {}
                # Store Dictionary of letters {0:[(embedding,letter_idx,preds),
                train_letters = {}
                master_index = {}
                for idx in range(self.len):
                    k = labels[idx].item()
                    # Stores the embedding(x), label(y), and outputd(z)
                    if k not in train_letters.keys():
                        train_letters[k]=[]
                    train_letters[k].append((embeddings[idx], k, preds_vector[idx], images[idx], idx))
                train_test_letters.append(train_letters)

                # Be able to reference a specific embedding from idx
                # train_test_master_index["test"][1236] = char, idx
                train_test_master_index[var] = master_index

            train_letters, test_letters = train_test_letters
            mean,std = self.compute_average(train_letters)  # d["train"]

            d = {"train":train_test_letters[0],
                "test": train_test_letters[1],
                "indices": train_test_master_index,
                "mean": mean,
                "std": std
                 }

            torch.save(d, save_path)
            # Verify different number of examples (e.g., first char 1)
            assert len(train_letters[1]) != len(test_letters[1])
        else:
            d = torch.load(save_path)
        self.setup_from_dict(d)

    def setup_from_dict(self, d):
        self.embd_mean, self.embd_std = d["mean"],d["std"]

        self.master_index = d["indices"]
        if self.which == 'train':
            self.letters = d["train"]
        elif self.which == 'test':
            self.letters = d["test"]

    def compute_average(self, letters):
        combined = []
        for i in letters.values():
            combined.extend([f[0] for f in i])
        self.embd_std, self.embd_mean = torch.std_mean(torch.stack(combined), 0)
        return self.embd_mean, self.embd_std

    def __getitem__(self, idx):
        char, emb_idx = self.master_index[self.which]
        return self.letters[char][emb_idx]

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


def collate_fn_embeddings(data):
    """
                          alphabet_size=27,
                          sentence_length=32,
                          embedding_dim=512
    Args:
        data (list): list of dicts of length (batch)

    Returns:

    """
    keys = data[0].keys()
    output_dict = {}
    padding = {"masked_gt": -100,
                "attention_mask": 0,
                #"data": 0,
                "gt_idxs": 0,
                "vgg_logits": 0,
                "embedding":0,
                "image": -0.4242}

    for data_key in keys:
        batch_data = [b[data_key] for b in data]
        if data_key in padding.keys() and batch_data and torch.is_tensor(batch_data[0]):
            #print(data_key, type(batch_data[0]))
            batch_data = torch.nn.utils.rnn.pad_sequence(batch_data, batch_first=True, padding_value=padding[data_key])
        if data_key == "num_preds":
            output_dict["num_preds"] = np.sum(batch_data) # masked_gt[masked_gt != -100].shape[0]
        else:
            output_dict[data_key] = batch_data
    return output_dict

def createSentenceDataset(sen_list_path,
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

    # sentence_data = load_sen_list(sen_list_path)
    sentence_data = None

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
    print("Creating Sentence Dataset...")
    SAVE_PATH = save_folder / "train_test_sentenceDataset.pt"
    createSentenceDataset(sen_list_path=ROOT / 'data/text_generation/sent_list.pkl',
                          emnist_sampler_path_train=save_folder / 'train_sorted_emnist.pt',
                          emnist_sampler_path_test=save_folder / 'test_sorted_emnist.pt',
                          embedding_sampler_path_train=save_folder / 'train_test_emb_dict.pt',
                          embedding_sampler_path_test=save_folder / 'train_test_emb_dict.pt',
                          save_path=SAVE_PATH)
    #obj = SentenceDataset(SAVE_PATH)

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

def load_sen_dataset(*args, **kwargs):
    sd = SentenceDataset(ROOT / (FOLDER + 'train_test_sentenceDataset.pt'), *args, **kwargs)
    return sd

def get_inputs(text,
               corpus,
               mask_id,
               mask_index=None):
    """ Train - with no mask
              - with mask

        Masking Values:
              -100 (labels) : will not be predicted
              0 (input) : will not be considered in prediction

    Args:
        text:
        mask_id: the labels are just a list of indices, e.g. [1,12,38,...]; this is the mask's ID
        mask_index (int): None - no mask, -1 - random mask, [0,n] - the index of the sequence to mask

    Returns:

    """

    n = len(text)
    ids = []
    for char in text:
        ids.append(corpus.index(char))
    input_ids = torch.tensor(ids).unsqueeze(0)
    attention_mask = torch.ones([1,n])

    if not mask_index is None:
        mask_index = np.random.randint(0, n) if mask_index < 0 else mask_index
        attention_mask[0, mask_index] = 0 # mask the input to be predicted

    labels = copy.deepcopy(input_ids)

    if not mask_index is None:
        # input_ids[0][mask_index] = mask_id  # character to predict is set to the vocab size????
        labels[input_ids != mask_id] = -100 # set everything else to not predict
    labels = labels.type(torch.LongTensor)
    return input_ids, attention_mask, labels, mask_index


def get_text(one_hot_tensor):
    """ Converts either one-hot tensor or index tensor to letters

    Args:
        tnsr:

    Returns:

    """
    text = ''
    for value in one_hot_tensor:
        if value.dim() and len(value) > 1:  # accepts either one-hot tensor OR one digit tensor
            value = torch.argmax(value)
        c = num_to_letter(value.item())
        text = text + c
    return text

def make_testing_version(*args,**kwargs):
    print("Making testing version")
    sd = SentenceDataset(ROOT / (FOLDER + 'train_test_sentenceDataset.pt'), *args, **kwargs)
    sd.save_small(ROOT / (FOLDER + 'train_test_sentenceDataset_SMALL.pt'))
    return sd

# Run Example
if __name__ == '__main__':
    #example_sen_loader()
    #import importlib, sen_loader
    #importlib.reload(sen_loader); from sen_loader import *
    RESET=False

    # Create datasets
    sd_full = create_datasets(ROOT / FOLDER,
                    ROOT / (FOLDER + "train_emb_dataset.pt"),
                    ROOT / (FOLDER + "test_emb_dataset.pt")
                    )

    # Test loading it
    sd = make_testing_version(which='Embeddings', train_mode="multicharacter MASK_CHAR")
    m = next(iter(sd))



"""
import importlib, sen_loader;importlib.reload(sen_loader); from sen_loader import *
sd = sen_loader.SentenceDataset('train_test_sentenceDataset.pt')
sd = load_sen_loader()
m = next(iter(sd))

train_loader = torch.utils.data.DataLoader(sd, batch_size=2, shuffle=True, collate_fn=collate_fn_embeddings)
n = next(iter(train_loader))
n["data"].shape

num_to_letter(torch.argmax(m[-2], axis=1))
"""
