#! /usr/bin/env python

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class EncodingDataset(Dataset):

    def __init__(self, encoding_pkl, label_pkl, ignore_enc, transform=None):
        super(EncodingDataset, self).__init__()

        self.transform = transform

        self.encodings = []
        self.labels = []
        self.ignore_enc = []

        with open(encoding_pkl, 'rb') as ph:
            self.encodings = pickle.load(ph)
        with open(label_pkl, 'rb') as ph:
            self.labels = pickle.load(ph)
        with open(ignore_enc, 'rb') as ph:
            self.ignore_enc = pickle.load(ph)

        class_labels_tmp = list(set(self.labels))

        min_num_examples = 5
        valid_idcs = []
        outcasts = []
        for l in class_labels_tmp:
            idcs = np.where(np.array(self.labels) == l)[0]
            if len(idcs) >= min_num_examples:
                valid_idcs.extend(idcs)
            else:
                outcasts.extend(idcs)

        # Get rid of classes with very few instances
        outcast_encs = [self.encodings[i] for i in outcasts]
        self.ignore_enc.extend(outcast_encs)
        self.encodings = [self.encodings[i] for i in valid_idcs]
        self.labels = [self.labels[i] for i in valid_idcs]

        self.labels = np.array(self.labels)
        # Re-init class labels
        self.class_labels = list(set(self.labels))

        assert len(self.labels) == len(self.encodings)


    def __len__(self):
        # return len(self.labels)
        return int(1e6 * 64 + 1e3)


    def __getitem__(self, index):
        if index % 2 == 1: # Same class
            label = 1.0
            class_idx1 = random.choice(self.class_labels)
            labels_to_choose = np.where(self.labels == class_idx1)[0]
            enc1 = self.encodings[random.choice(labels_to_choose)]
            enc2 = self.encodings[random.choice(labels_to_choose)]
        else:
            label = 0.0
            class_idx1 = random.choice(self.class_labels)
            labels_to_choose_1 = np.where(self.labels == class_idx1)[0]
            class_idx2 = random.choice(self.class_labels)
            labels_to_choose_2 = np.where(self.labels == class_idx2)[0]
            enc1 = self.encodings[random.choice(labels_to_choose_1)]
            if random.random() < 0.3:
                enc2 = random.choice(self.ignore_enc)
            else:
                enc2 = self.encodings[random.choice(labels_to_choose_2)]

        enc1 = torch.Tensor(enc1)
        enc2 = torch.Tensor(enc2)

        # if self.transform:
        #     enc1 = self.transform(enc1)
        #     enc2 = self.transform(enc2)

        assert torch.abs(torch.max(enc1)) < 1
        assert torch.abs(torch.max(enc2)) < 1

        return enc1, enc2, label

class TestEncodingDataset(EncodingDataset):
    def __init__(self, encoding_pkl, label_pkl, ignore_enc, transform=None):
        super(TestEncodingDataset, self).__init__(encoding_pkl, label_pkl, ignore_enc, transform)
    
    def __getitem__(self, index):
        choice = random.random() < 0.5

        idx_label = self.labels[index]
        # Same label
        if choice: 
            ignore_choice = False
            labels_to_choose = np.where(self.labels == idx_label)[0]
            label = 1.0
        else:
            ignore_choice = random.random() < 0.3
            labels_to_choose = np.where(self.labels != idx_label)[0]
            label = 0.0

        if len(labels_to_choose) <= 1:
            raise NotImplementedError("uh-oh")
        if ignore_choice:
            enc_2 = torch.Tensor(random.choice(self.ignore_enc))
            # print('ignore')
        else:
            other_idx = random.choice(labels_to_choose)
            enc_2 = torch.Tensor(self.encodings[other_idx])
            # print(index, other_idx)

        enc_1 = torch.Tensor(self.encodings[index])

        if self.transform:
            enc_1 = self.transform(enc_1)
            enc_2 = self.transform(enc_2)

        return enc_1, enc_2, label

    def __len__(self):
        return(len(self.labels))



class encodingSiamese(nn.Module):

    def __init__(self, encoding_size, hidden_size):
        super(encodingSiamese, self).__init__()

        # self.fc1 = nn.Linear(encoding_size, 64)
        self.intermediate_size = hidden_size
        self.linear = nn.Sequential(nn.Linear(encoding_size, self.intermediate_size), nn.Sigmoid())
        self.out = nn.Linear(self.intermediate_size, 1)

    def forward_one(self, x):
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        distance = torch.abs(out1 - out2)
        out = self.out(distance)

        return out


if __name__ == '__main__':
    # long_dataset = EncodingDataset('/code/MISC_DATA/long_enc_2018.pkl', '/code/MISC_DATA/id_nums_2018.pkl', '/code/MISC_DATA/long_enc_IGNORE.pkl')

    enc_type = 'short'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dir = '/code/MISC_DATA'

    train_sets = []
    for year in range(2018, 2020):
        print(year)
        tmp_set = EncodingDataset(os.path.join(root_dir, f'{enc_type}_enc_{year}.pkl'), os.path.join(root_dir, f'id_nums_{year}.pkl'), os.path.join(root_dir, f'{enc_type}_enc_IGNORE.pkl'))
        train_sets.append(tmp_set)
        # train_set = EncodingDataset(os.path.join(root_dir, 'short_enc_2018.pkl'), os.path.join(root_dir, 'id_nums_2018.pkl'), os.path.join(root_dir, 'short_enc_IGNORE.pkl'))
        # train_set = EncodingDataset(os.path.join(root_dir, 'long_enc_2018.pkl'), os.path.join(root_dir, 'id_nums_2018.pkl'), os.path.join(root_dir, 'long_enc_IGNORE.pkl'))

    if enc_type == 'short':
        input_size = 128
        hidden_size = 256
        test_set = TestEncodingDataset(os.path.join(root_dir, 'short_enc_2020.pkl'), os.path.join(root_dir, 'id_nums_2020.pkl'), os.path.join(root_dir, 'short_enc_IGNORE.pkl'))
    elif enc_type == 'long':
        input_size = 512
        hidden_size = 256
        test_set = TestEncodingDataset(os.path.join(root_dir, 'long_enc_2020.pkl'), os.path.join(root_dir, 'id_nums_2020.pkl'), os.path.join(root_dir, 'long_enc_IGNORE.pkl'))

    train_loaders = []
    for year_set in train_sets:
        tmpLoader = enumerate(DataLoader(year_set, batch_size=64, shuffle=True))
        train_loaders.append(tmpLoader)

    testLoader = DataLoader(test_set, batch_size=64, shuffle=True)

    # Set up training
    lr = 0.0004
    max_iter = 10000000

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = encodingSiamese(input_size, hidden_size).to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20000, T_mult = 2)
    train_loss = []
    loss_val = 0
    time_start = time.time()
    # queue = deque(maxlen=20)

    thresh=0.3

    def run_test(train_batch_num):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(10):
                for bnum, (tEnc1, tEnc2, tLabel) in enumerate(testLoader, 1):
                    tEnc1 = tEnc1.to(device)
                    tEnc2 = tEnc2.to(device)
                    tLabel = tLabel.to(device).reshape(-1, 1)
    
                    tOutput = net(tEnc1, tEnc2)
                    sig = nn.Sigmoid()
                    tOutput = sig(tOutput)
                    diff = torch.abs(tOutput - tLabel)
                    batch_correct = len(torch.where(diff < thresh)[0])
    
                    total += len(tLabel)
                    correct += batch_correct

        print(f"TEST RUN {train_batch_num}: {correct}/{total}: {correct/total*100:.2f}%")

        net.train()


    for itnum in range(1000000):
        for tmp in train_loaders:
            (enc1, enc2, label) = next(tmp)
            enc1 = Variable(enc1.to(device))
            enc2 = Variable(enc2.to(device))
            label = Variable(label.to(device)).reshape(-1, 1)

            optimizer.zero_grad()
            output = net(enc1, enc2)
            loss = loss_fn(output, label)
            # loss_val += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if batch_id % 100 == 0:
            print(batch_id, loss.item(), scheduler.get_last_lr())
        if batch_id % 10000 == 0:
            run_test(batch_id)
            torch.save(net.state_dict(), f'weights_{enc_type}_{batch_id}.ptw')
