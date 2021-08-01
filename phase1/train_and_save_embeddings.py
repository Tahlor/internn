# CONFIG
# CNN
# LM
import pdb
import torch.nn.functional as FN
from pathlib import Path
import os
import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
import sys
import wandb
from general_tools.utils import get_root
ROOT = get_root("internn")

os.chdir("..")
sys.path.append(os.path.abspath("./data"))
#import data.sen_loader
from data.sen_loader import collate_fn, save_dataset, SentenceDataset

from models.VGG import *
import argparse
from data import loaders
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

OUTPUT = ROOT / "data/embedding_datasets/embeddings_v2"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
    wandb.init(project="TEST")

    return opts

device = 'cuda'


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test_model_load():
    """ Initialized model test set accuracy: Less than 5%
        Loaded model test set accuracy: usually around 95.3894%"""
    num_epochs = 200
    learning_rate = .007
    train_loader, test_loader = loaders.loader(batch_size_train=100, batch_size_test=1000)

    #model1 = VGG().to(device)
    model1 = VGG_embedding().to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    #Check test accuracy before load
    model1.eval()
    with torch.no_grad():  # turn off gradient calc
        correct1 = 0
        total1 = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

    loadVGG(model1)

    # Test the model
    model1.eval()
    with torch.no_grad():  # turn off gradient calc
        correct1 = 0
        total1 = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

    exit()

def calc_embeddings(save_folder=OUTPUT, embedding=True):
    """
    Save the embeddings out
        embedding (bool): True - save the embedding; False - save the softmax
    Returns:

    """
    save_folder = Path(save_folder); save_folder.mkdir(exist_ok=True, parents=True)
    train_loader, test_loader = loaders.loader(batch_size_train=100, batch_size_test=1000)
    model1 = VGG_embedding().to(device)
    loadVGG(model1, path=OUTPUT / "vgg.pt")

    for l in "train","test":
        loader = train_loader if l == "train" else test_loader
        # Test the model
        model1.eval()
        embeddings_list = None
        labels_list = None
        one_hot_pred_list = None

        with torch.no_grad():  # turn off gradient calc
            first = True
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                embeddings = model1.get_embedding(images)
                pred = model1.forward_embedding(embeddings) # this will output the softmax

                if first:
                    embeddings_list = embeddings
                    labels_list = labels
                    one_hot_pred_list = pred
                    first = False
                else:
                    embeddings_list = torch.cat((embeddings_list, embeddings), 0)  # torch.Size([100, 512])
                    labels_list = torch.cat((labels_list, labels), 0)  # torch.Size([100])
                    one_hot_pred_list = torch.cat((one_hot_pred_list, pred), ) # torch.Size([100, 27])

            # Calculate space
            space_image = (torch.zeros(*images.shape) + torch.min(images).tolist()).to(device)[0:1]
            label = torch.zeros([1]).to(device)
            label[0] = 0
            embd = model1.get_embedding(space_image)
            pred = model1.forward_embedding(embd)

            embeddings_list = torch.cat((embeddings_list, embd), 0)
            labels_list = torch.cat((labels_list, label), 0)
            #pred = torch.zeros(one_hot_pred_list.shape[1]).to(device); pred[0] = 1
            #pred = FN.one_hot(0, num_classes=one_hot_pred_list.shape[1]) # NO YOU NEED TO GET A PREDICTION THAT GOES TO THE SPACE YOU IDIOT
            one_hot_pred_list = torch.cat((one_hot_pred_list, pred), 0)

        # Save new calculated embeddings one hot encoded
        torch.save((embeddings_list, labels_list, one_hot_pred_list), save_folder / f'{l}_emb_dataset.pt')

def main(num_epochs = 200,
         learning_rate = 0.005,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    # Loads random characters
    train_loader, test_loader = loaders.loader(batch_size_train = 100, batch_size_test = 1000)

    # Train the model
    total_step = len(train_loader)
    curr_lr1 = learning_rate

    #model1 = VGG().to(device)
    model1 = VGG_embedding().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    best_accuracy1 = 0
    best_model = None

    space_image = (torch.zeros(28,28) + -.4242)[None,None,:,:].to(device)

    for epoch in range(num_epochs):
        model1.train()
        for i, (images, labels) in enumerate(train_loader): #Modify if lengths need to be returned from collate fn
            images = images.to(device)
            images = torch.cat([images, space_image], axis=0)
            labels = torch.cat([labels,torch.zeros(1).int()])
            labels = labels.to(device)
            # Forward
            outputs = model1(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            optimizer1.zero_grad() # clears old gradients from the previus step
            loss1.backward() # computes the derivative of the loss w,r,t the params using back propogation
            optimizer1.step() # causes the optimizer to take a step based on the gradients of the params (updates the weights)

            if i == 499:
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item()))

        # Test the model
        model1.eval() # turns off dropout layers and batchnorm layers, aka sets model in inference mode

        with torch.no_grad(): # turn off gradient calc
            correct1 = 0
            total1 = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model1(images)
                _, predicted = torch.max(outputs.data, 1) # returns vals and indices with dim to reduce = 1(cols)
                total1 += labels.size(0)
                correct1 += (predicted == labels).sum().item()

            if best_accuracy1 >= correct1 / total1:
                curr_lr1 = learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer1, curr_lr1)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
            else:
                best_accuracy1 = correct1 / total1
                net_opt1 = model1
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

                # Save best model - Comment out if you intend on using the supercomputer to only write the best model after training
                best_model = model1
                saveVGG(model1, path=OUTPUT / "vgg.pt")


    #saveVGG(best_model) # Uncomment for use on the fsl otherwise to get a saved model training needs to run to finish
    calc_embeddings(OUTPUT)

def resave():
    calc_embeddings(OUTPUT)


if __name__=='__main__':
    #main()
    resave()
