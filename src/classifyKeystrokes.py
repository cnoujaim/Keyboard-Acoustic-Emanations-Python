
import sys
import os
import datetime
import re
import sklearn
import torch
import json


import numpy as np
from sklearn.cluster import KMeans
from librosa.feature import mfcc

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class KeyDataLoader(Dataset):
    """Dataloader."""

    def __init__(self, files):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.keystrokes = []
        for file in files:
            with open(file, "r") as f:
                data = json.loads(f.read().strip('\n'))
                data = [(self.extract_features(np.array(a)),  self.convertletter(b)) for (a, b) in data]
                self.keystrokes.extend(data)

    def __len__(self):
        return len(self.keystrokes)

    def __getitem__(self, idx):
        return self.keystrokes[idx]

    def convertletter(self, l):
        i = ord(l) - ord('a')
        if i < 0 or i > 25:
            i = 26
        y = torch.zeros(27)
        y[i] = 1
        return y

    def extract_features(self, keystroke, sr=44100, n_mfcc=16, n_fft=441, hop_len=110):
        '''Return an MFCC-based feature vector for a given keystroke.'''
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft, # n_fft=220 for a 10ms window
                    hop_length=hop_len, # hop_length=110 for ~2.5ms
                    )
        return torch.tensor(spec.flatten())


class KeyNet(nn.Module):
    def __init__(self):
        super(KeyNet, self).__init__()
        self.fc1 = nn.Linear(336, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 27)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifyKeystrokes:
    '''Classifies keystrokes using clustering and HMMs'''
    def __init__(self, files):
        self.dataset = KeyDataLoader(files)
        self.dataloader = DataLoader(self.dataset, shuffle=True)
        self.net = KeyNet()
        self.classify()

    def classify(self):
        '''Classify keystrokes'''
        print("Training neural network...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(100):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')



def main():
    '''Run learn keystrokes'''
    files = sys.argv[1:]
    outfile = os.path.join("out", "raw_sentence", str(datetime.datetime.now()) + "_raw")
    print("--- Classify Keystrokes ---")

    classifier = ClassifyKeystrokes(files)



main()
