
import sys
import os
import datetime
import re
import sklearn
import torch
import json
import copy
import datetime

import numpy as np
from sklearn.cluster import KMeans
from librosa.feature import mfcc

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
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

        # self.keystrokes = [i for i in self.keystrokes if i[1] > 15]


    def __len__(self):
        return len(self.keystrokes)

    def __getitem__(self, idx):
        return self.keystrokes[idx]

    def convertletter(self, l):
        i = ord(l) - ord('a')
        if i < 0 or i > 25:
            i = 26
        return torch.tensor(i).long()

    def extract_features(self, keystroke, sr=44100, n_mfcc=16, n_fft=220, hop_len=110):
        '''Return an MFCC-based feature vector for a given keystroke.'''
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft, # n_fft=220 for a 10ms window
                    hop_length=hop_len, # hop_length=110 for ~2.5ms
                    )
        return torch.tensor(spec.flatten())

def convertnumber(n):
    i = chr(n + ord('a'))
    if n < 0 or n > 25:
        i = " "
    return i

class KeyNet(nn.Module):
    def __init__(self):
        super(KeyNet, self).__init__()
        self.fc1 = nn.Linear(1296, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 27)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifyKeystrokes:
    '''Classifies keystrokes using clustering and HMMs'''
    def __init__(self, files, shuffle=True):
        self.freq_letters = [26, 4, 19, 0, 14, 8, 13, 18, 17, 7, 11, 3, 2, 20, 12, 5, 15, 6, 22, 24, 1, 21, 10, 23, 9, 16, 25]
        self.dataset = KeyDataLoader(files)
        validation_split = .2

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle == True:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        self.trainloader = DataLoader(self.dataset, sampler=train_sampler)
        self.validloader = DataLoader(self.dataset, sampler=valid_sampler)
        self.net = KeyNet()
        class3 = "out/raw_sentence/models/fc_nn_model_10:51:20.187248.txt"
        if os.path.exists(class3):
            self.net.load_state_dict(torch.load(class3))
        else:
            self.classify()
            self.savemodel()

        valid = KeyDataLoader(["out/keystrokes/typingpractice2.wav_out"])
        self.validloader = DataLoader(valid)
        self.validate_sentence()

    def classify(self):
        '''Classify keystrokes'''
        print("Training neural network...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.000005)
        best_acc = 0
        best_model = None
        for epoch in range(500):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
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
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
            running_acc = self.validate()
            if running_acc > best_acc:
                best_acc = running_acc
                best_model = copy.deepcopy(self.net)

        self.net = best_model
        print('Finished Training')

    def validate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.validloader:
                images, labels = data
                outputs = self.net(images)
                # print(images)
                # print(f"outputs {outputs}")
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += 1
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Accuracy of the network on the {len(self.validloader)} test keys: {acc}%')
        return acc

    def validate_sentence(self):
        correct = 0
        total = 0
        sentence = ""
        full_labels = []
        with torch.no_grad():
            for data in self.validloader:
                images, labels = data
                outputs = self.net(images)

                full_labels.append(outputs)

                _, predicted = torch.max(outputs.data, 1)
                sentence += convertnumber(predicted)
                total += 1
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Accuracy of the network on the {len(self.validloader)} test keys: {acc}%')
        print(f"Predicted sentence:")
        print(sentence)
        print("")
        print("Apply corrections")
        full_labels = np.stack(full_labels, axis=0)
        final = self.correct_with_prob(full_labels)
        return acc

    def savemodel(self):
        path = f"out/raw_sentence/models/fc_nn_model_{datetime.datetime.now().time()}.txt"
        print(f'Saving model in {path}')
        torch.save(self.net.state_dict(), path)

    def correct_with_prob(self, labels):
        sentence = []
        cur_word = []
        for i in range(len(labels)):
            if (convertnumber(np.argmax(labels[i])) == " "):
                sentence.append(cur_word)
                cur_word = []
            else:
                top_indices = np.flip(np.argsort(labels[i])[0])[:4]
                # print(top_indices)
                top_letters = [convertnumber(i) for i in top_indices]
                cur_word.append(top_letters)


        sentence.append(cur_word)

        with open('out/raw_sentence/out_text.json', 'w') as f:
            json.dump(sentence, f)


def main():
    '''Run learn keystrokes'''
    files = sys.argv[1:]
    outfile = os.path.join("out", "raw_sentence", str(datetime.datetime.now()) + "_raw")
    print("--- Classify Keystrokes ---")

    classifier = ClassifyKeystrokes(files)



main()
