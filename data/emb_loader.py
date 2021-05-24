import torch
import errno
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)


class VggEmbeddingsDataset(Dataset):
    def __init__(self, PATH):
        self.x = None
        self.y = None
        self.len = 0
        try:
            dataset = torch.load(PATH)
            self.x = dataset[0]
            self.y = dataset[1]
            self.len = len(self.y)
        except FileNotFoundError:
            print("File not found. Check path to File.")



        # load embeddings_dataset.pt
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def save_dataset(self, x, y, PATH):
        one_hot_labels = F.one_hot(y)
        torch.save((x, one_hot_labels), PATH)

def example_emb_loader():
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
