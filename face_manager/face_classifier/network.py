#! /usr/bin/env python 

import torch.nn as nn
import torch.nn.functional as F


class FaceNetwork(nn.Module):
    def __init__(self, n_classes, dim_hidden, which_features='short'):
        super(FaceNetwork, self).__init__()
        assert which_features in ['short', 'long']
        
        # Input is 128-dimensional vector
        if which_features == 'short':
            self.fc1 = nn.Linear(128, 400)
            self.fc2 = nn.Linear(400, 750)
        else:
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 750)
        self.nl1 = nn.LeakyReLU(.05)
        self.d1 = nn.Dropout(p = 0.2)
        self.nl2 = nn.LeakyReLU(.05)
        self.d2 = nn.Dropout(p = 0.2)
        self.fc3 = nn.Linear(750, 512)
        self.d3 = nn.Dropout(p = 0.2)
        self.nl3 = nn.Tanh()
        # self.nl3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(512, dim_hidden)
        # self.fc5 = nn.Linear(dim_hidden, n_classes)
        self.fc5 = nn.Linear(750, n_classes)

    def forward(self, x):
 #       x = self.d1(x)
        x = self.fc1(x)
        x = self.nl1(x)
#        x = self.d2(x)
        x = self.fc2(x)
        x = self.nl2(x)
       # x = self.d3(x)
       # x = self.fc3(x)
       # x = self.nl3(x)
       # hidden_out = self.fc4(x)
       # logits = self.fc5(hidden_out)
        logits = self.fc5(x)

       # return hidden_out, logits, F.softmax(logits, dim=1)
        return None, logits, F.softmax(logits, dim=1)
