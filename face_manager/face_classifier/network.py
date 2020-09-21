#! /usr/bin/env python 

import torch.nn as nn
import torch.nn.functional as F


class FaceNetwork(nn.Module):
    def __init__(self, n_classes, dim_hidden):
        super(FaceNetwork, self).__init__()
        
        # Input is 128-dimensional vector
        self.fc1 = nn.Linear(128, 256)
        self.nl1 = nn.LeakyReLU(.05)
        self.fc2 = nn.Linear(256, 350)
        self.nl2 = nn.LeakyReLU(.05)
        self.fc3 = nn.Linear(350, 512)
        self.nl3 = nn.Tanh()
        # self.nl3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(512, dim_hidden)
        self.fc5 = nn.Linear(dim_hidden, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nl1(x)
        x = self.fc2(x)
        x = self.nl2(x)
        x = self.fc3(x)
        x = self.nl3(x)
        hidden_out = self.fc4(x)
        logits = self.fc5(hidden_out)

        return hidden_out, logits, F.softmax(logits, dim=1)
