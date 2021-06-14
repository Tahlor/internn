# CONFIG
# CNN
# LM
import pdb
import os
import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
from data import emb_loader
import sys

from data.emb_loader import VggEmbeddingsDataset

sys.path.append(os.path.abspath("./data"))
import data.sen_loader
#from data.sen_loader import collate_fn, SentencesDataset
from models.VGG import *
import argparse
from data import loaders
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/stroke_config/baseline.yaml", help='Path to the config file.')
    parser.add_argument('--testing', action="store_true", default=False, help='Run testing version')
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

def calc_embeddings():
    num_epochs = 200
    learning_rate = .007
    train_loader, test_loader = loaders.loader(batch_size_train=100, batch_size_test=1000)

    model1 = VGG_embedding().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    loadVGG(model1)

    # Test the model
    model1.eval()
    data = None
    labs = None
    preds = None
    with torch.no_grad():  # turn off gradient calc
        correct1 = 0
        total1 = 0
        x = list()
        y = list()
        z = list()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # # Calc embeddings for a space ' '

            # images = np.full((1, 28, 28), fill_value=-0.4242, dtype=np.float32)
            # images = torch.from_numpy(images)
            # images = torch.unsqueeze(images, 0)
            # labels = torch.tensor([0])
            # labels = labels.to(device)
            # images = images.to(device)

            embeddings = model1.get_embedding(images)
            pred = model1.classifier(embeddings)
            x.append(embeddings)
            y.append(labels)
            z.append(pred)


        for i in range(len(x)):
            if i == 0:
                data = x[i]
                labs = y[i]
                preds = z[i]
            else:
                data = torch.cat((data, x[i]), 0)
                labs = torch.cat((labs, y[i]), 0)
                preds = torch.cat((preds, z[i]), )

    # Initialize embeddings dataset
    embedd1 = VggEmbeddingsDataset('./emb_dataset.pt')
    # Save new calculated embeddings
    embedd1.save_dataset(data, labs, preds, 'emb_dataset_train.pt')

    # Calculate Testset embeddings
    model1 = VGG_embedding().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    loadVGG(model1)
    model1.eval()
    data = None
    labs = None
    preds = None
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        x = list()
        y = list()
        z = list()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # # Calc embeddings for a space ' '

            # images = np.full((1, 28, 28), fill_value=-0.4242, dtype=np.float32)
            # images = torch.from_numpy(images)
            # images = torch.unsqueeze(images, 0)
            # labels = torch.tensor([0])
            # labels = labels.to(device)
            # images = images.to(device)

            embeddings = model1.get_embedding(images)
            pred = model1.classifier(embeddings)
            x.append(embeddings)
            y.append(labels)
            z.append(pred)

        for i in range(len(x)):
            if i == 0:
                data = x[i]
                labs = y[i]
                preds = z[i]
            else:
                data = torch.cat((data, x[i]), 0)
                labs = torch.cat((labs, y[i]), 0)
                preds = torch.cat((preds, z[i]), )

    # Save new calculated embeddings
    embedd1.save_dataset(data, labs, preds, 'test_emb_dataset.pt')

    exit()
    # save


def main(num_epochs = 200,
         learning_rate = 0.005,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    calc_embeddings()
    #test_model_load()

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

    ### VISUAL PART

    # Create model
    # Sample letters
    # Choose random images of letters
    # Restrict language model to predict only A-z
    # Pre train the language model
    # [Calculate statistics on visual model only OR train it]

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): #Modify if lengths need to be returned from collate fn
            images = images.to(device)
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
                saveVGG(model1)

            model1.train()
    #saveVGG(best_model) # Uncomment for use on the fsl otherwise to get a saved model training needs to run to finish


if __name__=='__main__':
    main()

