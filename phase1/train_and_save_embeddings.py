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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from general_tools.utils import get_root
from internn_utils import process_config

ROOT = get_root("internn")
print(os.getcwd())
os.chdir("..")
sys.path.append(os.path.abspath("./data"))
#import data.sen_loader

from models.VGG import *
import argparse
from data import loaders
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / "phase1/configs/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
    parser.add_argument('--calc_embeddings', action="store_true", default=False, help='Calc embeddings only')
    #parser.add_argument('--name', type=str, default="", help='Optional - special name for this run')
    opts = parser.parse_args()
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

def calc_embeddings(config):
    """
    """
    save_folder = Path(config.save_folder); save_folder.mkdir(exist_ok=True, parents=True)
    train_loader, test_loader = loaders.loader(batch_size_train=100, batch_size_test=1000)
    model1 = VGG_embedding().to(device)
    loadVGG(model1, path= config.save_folder / "vgg.pt")

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
                pred = model1.forward_embedding(embeddings).cpu() # this will output the softmax

                if first:
                    embeddings_list = embeddings.cpu()
                    labels_list = labels
                    one_hot_pred_list = pred
                    first = False
                else:
                    embeddings_list = torch.cat((embeddings_list, embeddings.cpu()), 0)  # torch.Size([100, 512])
                    labels_list = torch.cat((labels_list, labels), 0)  # torch.Size([100])
                    one_hot_pred_list = torch.cat((one_hot_pred_list, pred), ) # torch.Size([100, 27])

            # Calculate space
            space_image = (torch.zeros(*images.shape) + torch.min(images).tolist()).to(device)[0:1]
            label = torch.zeros([1]).to(device)
            label[0] = 0
            embd = model1.get_embedding(space_image)
            pred = model1.forward_embedding(embd).cpu()

            embeddings_list = torch.cat((embeddings_list, embd.cpu()), 0)
            labels_list = torch.cat((labels_list, label), 0)
            #pred = torch.zeros(one_hot_pred_list.shape[1]).to(device); pred[0] = 1
            #pred = FN.one_hot(0, num_classes=one_hot_pred_list.shape[1]) # NO YOU NEED TO GET A PREDICTION THAT GOES TO THE SPACE YOU IDIOT
            one_hot_pred_list = torch.cat((one_hot_pred_list, pred), 0)

        # Save new calculated embeddings one hot encoded
        # if "normalized" in str(OUTPUT):
        #     for i in range(0,len(embeddings_list)):
        #         embeddings_list[i] = torch.nn.functional.normalize(embeddings_list[i], dim=-1)

        torch.save((embeddings_list, labels_list, one_hot_pred_list), save_folder / f'{l}_emb_dataset.pt')

def main(config,
         *args,
         **kwargs):

    # Loads random characters
    train_loader, test_loader = loaders.loader(batch_size_train=config.batch_size_train,
                                               batch_size_test=config.batch_size_test)

    # Train the model
    total_step = len(train_loader)
    curr_lr1 = config.learning_rate

    #model1 = VGG().to(device)
    model1 = VGG_embedding().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer1, 'min', patience=config.patience, factor=config.decay_factor)

    # Train the model
    total_step = len(train_loader)

    best_accuracy1 = 0
    best_model = None

    space_image = (torch.zeros(28,28) + -.4242)[None,None,:,:].to(device)

    for epoch in range(config.epochs):
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
            #scheduler.step(loss1)
            if i == 499:
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, config.epochs, i + 1, total_step, loss1.item()))
            if config.TESTING:
                break

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
                curr_lr1 = config.learning_rate * np.asscalar(pow(np.random.rand(1), 3))
                update_lr(optimizer1, curr_lr1)
                print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100 * best_accuracy1))
            else:
                best_accuracy1 = correct1 / total1
                net_opt1 = model1
                print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))

                # Save best model - Comment out if you intend on using the supercomputer to only write the best model after training
                best_model = model1
                saveVGG(model1, path=Path(config.save_folder) / "vgg.pt")

    #saveVGG(best_model) # Uncomment for use on the fsl otherwise to get a saved model training needs to run to finish

def resave():
    calc_embeddings(config.save_folder)

if __name__=='__main__':
    opts = parse_args()
    config = process_config(opts.config)
    Path(config.save_folder).mkdir(parents=True, exist_ok=True)
    if config.wandb:
        wandb.init(project="TEST")
    if not opts.calc_embeddings:
        main(config)
    resave()
