# CONFIG
# CNN
# LM
import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
from models.VGG import *
import argparse
from data import loaders

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

def main(num_epochs = 200,
         learning_rate = 0.005,
         momentum = 0.5,
         log_interval = 500,
         *args,
         **kwargs):

    train_loader, test_loader = loaders.loader(batch_size_train = 100, batch_size_test = 1000)

    # Train the model
    total_step = len(train_loader)
    curr_lr1 = learning_rate

    model1 = VGG().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)

    best_accuracy1 = 0

    ### VISUAL PART

    # Create model
    # Sample letters
    # Choose random images of letters
    # Restrict language model to predict only A-z
    # Pre train the language model
    # [Calculate statistics on visual model only OR train it]

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model1(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            if i == 499:
                print("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss1.item()))

        # Test the model
        model1.eval()

        with torch.no_grad():
            correct1 = 0
            total1 = 0

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model1(images)
                _, predicted = torch.max(outputs.data, 1)
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

            model1.train()

if __name__=='__main__':
    main()