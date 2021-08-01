# -*- coding: utf-8 -*-

import torch.optim as optim
import copy
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
from sen_loader import *
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import *
import sys
from CustomBert import *
sys.path.append("..")
from pytorch_utils import *
from general_tools.my_logging import *
from BERT_BASE import *

ROOT = get_root("internn")
print(ROOT)
PATH = ROOT / "data/embedding_datasets/embeddings_v2"

text = 'abcdefghi jklmnopqrstuvwxyz'
corpus = [char for char in text]
mask_id = len(corpus) + 1
vocab_size = mask_id + 2

train_dataset = SentenceDataset(PATH=PATH / 'train_test_sentenceDataset.pt', which='Embeddings')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = SentenceDataset(PATH=PATH / 'train_test_sentenceDataset.pt', which='Embeddings', train=False)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

vocab_size = 27
device = torch.device('cuda:0')

model = BertModelCustom(BertConfig(vocab_size=vocab_size + 2)).to(device)
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)

if PATH:
    latest = get_latest_file(PATH)
    print(f"Loading {latest}")
    model, optimizer, starting_epoch, loss = load_model(latest, model, optimizer)
    starting_epoch = 1

## TEST MODEL
input_ids, attention_mask, labels, mask_index = get_inputs("my name is sam")
output = model(input_ids.to(device), attention_mask.to(device))

total = 0
correct = 0

"""
Mask one letter and predict it
"""
for i_batch, sample in enumerate(train_loader):
    x, y_truth = sample[0].to(device), sample[1].to(device)

    text = get_text(y_truth.squeeze(0))
    input_ids, attention_mask, y_truth, index = get_inputs(text)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    output, y_hat = model(input_ids, attention_mask)

    if (text[index] == get_text(y_hat.argmax(-1))):
        correct = correct + 1

    total = total + 1

print(correct / total)

# correct = 0
# for x, y_truth in val_loader:
#   x, y_truth = x.to(device), y_truth.to(device)
#   output, y_hat = model(input_embeddings = x)
#   correct = correct + int(torch.sum(output.argmax(-1).squeeze(0) ==  y_truth.argmax(-1)))

# print(correct/len(val_loader.dataset))

total = 0
correct = 0
for i_batch, sample in enumerate(test_loader):
    x, y_truth = sample[0].to(device), sample[1].to(device)

    text = get_text(y_truth.squeeze(0))
    input_ids, attention_mask, y_truth, index = get_inputs(text)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    output, y_hat = model(input_ids, attention_mask)

    # print("Text: ", text, index)
    # print("Label: ", text[index])
    # print("Prediction: ", get_text(y_truth.argmax(-1)))

    if (text[index] == get_text(y_truth.argmax(-1))):
        correct = correct + 1

    total = total + 1

print(correct / total)
