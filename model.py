


#2. redo meta.py to meta1.py and meta2.py (first searching broadly and finding that populationsize s most important ,then fixing populationsize to 500 and finding that number of parents and crossover is best), then fixing all parameters and run for a long while in mnist.py make meta parameter search with batchsize~~ ca 2000 , nr of parent mating etc, population = 500




import numpy
import torch
import time
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
import argparse
import textwrap
from pygad import torchga
import pygad
import pandas
from pathlib import Path


def load_model(model_path,model):
    """
    :param model_path: path to saved model dictionary
    :param model pytorch model
    :return: the model loaded with the saved weights
    """
    model = torch.load(model_path)
    model.eval()
    return model

def save_model(solution,name,model,folder):
    """
    :param solution: the genome of the individual we want to save
    :param name: name to save state dict to
    :param folder: folder to save state dict in
    :param model pytorch model
    :return: None
    """
    # load the best weights and get the dictionary
    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)
    path_to_saved_model_dictionary = Path(folder)/Path(name)
    torch.save(model_weights_dict, path_to_saved_model_dictionary)


#pytorch implementation of the classic LeNet5 (copied from https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320)
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        #probs = F.softmax(logits, dim=1) #we use crossentropyloss wich includes a softmax inside the loss function
        #return logits, probs
        return logits



#a simple conv classifier
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output   # return x for visualization




