import random
import torch
import errno
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader


os.environ['CUDA_VISIBLE_DEVICES'] = str(0)


class VggEmbeddingsDataset(Dataset):
    def __init__(self, PATH=None, which='train'):
        self.x = None
        self.y = None
        self.z = None
        self.len = 0
        try:
            dataset = torch.load(PATH)
            self.x = dataset[0]
            self.y = dataset[1]
            self.z = dataset[2]
            self.len = len(self.y)
        except FileNotFoundError:
            print("File not found. Check path to File.")

        #self.emb_loaded = EmbbedingSampler(PATH, which)

    def __len__(self):
        return self.len

    def __getitem__(self, item):

        return self.x[item], self.y[item]

    # def loadSentences(self):
    #     train_dataset = SentencesDataset(os.path.abspath('data/sorted_train_emnist.pt'), 'train')
    #     train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=100, shuffle=False)
    def save_dataset(self, x, y, z, PATH):
        torch.save((x, y, z), PATH)
        # one_hot_labels = F.one_hot(y)
        # torch.save((x, one_hot_labels), PATH)


def example_emb_loader():
    #test = EmbbedingSampler('../emb_dataset.pt', 'train')
    emb_dataset = VggEmbeddingsDataset('../emb_dataset.pt')
    train_loader = torch.utils.data.DataLoader(emb_dataset, batch_size=100, shuffle=False)

    for i_batch, sample in enumerate(train_loader):
        if i_batch == 0:
            print(sample[0])
            print("Batch Data Shape = ", sample[0].shape)
            print("Num Batch Labels = ", sample[1].shape)
            break



# Example
#example_emb_loader()
