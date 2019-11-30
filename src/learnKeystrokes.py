import sys
import os
import datetime
import re
import sklearn
import json
import copy

import numpy as np
from sklearn.cluster import KMeans
from librosa.feature import mfcc

from hmmlearn import hmm

class ClassifyKeystrokes:
    '''Classifies keystrokes using clustering and HMMs'''
    def __init__(self, files):
        self.keystrokes = []
        for file in files:
            with open(file, "r") as f:
                data = json.loads(f.read().strip('\n'))
                data = [(self.extract_features(np.array(a)),  self.convertletter(b)) for (a, b) in data]
                self.keystrokes.extend(data)

        self.Xtrain = []
        self.ytrain = []
        self.convert_keystrokes_to_features()

        self.train()

    def train(self, iters=20):
        print("Training model...")
        best_model = None
        best_clus = None
        best_acc = 0
        for i in range(iters):
            print(f"-- Run iteration {i} --")
            clusters = self.cluster()
            clusters = self.Xtrain  # np.reshape(clusters, (-1, 1))
            model = self.hmm(clusters)
            print(f"Log probability for this model is {model.score(clusters)}")
            acc = self.accuracy(model, model.predict(clusters))
            print(f"Accuracy for this model is {acc}")
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                best_clus = copy.deepcopy(clusters)

        print(f"Prediction for this model is {self.print_predict(best_model.predict(best_clus))}")


    def extract_features(self, keystroke, sr=44100, n_mfcc=16, n_fft=441, hop_len=110):
        '''Return an MFCC-based feature vector for a given keystroke.'''
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft, # n_fft=220 for a 10ms window
                    hop_length=hop_len, # hop_length=110 for ~2.5ms
                    )
        return spec.flatten()


    def convert_keystrokes_to_features(self):
        '''Convert keystroke wav info to training information'''
        for keystroke, label in self.keystrokes:
            feat = self.extract_features(np.array(keystroke))
            self.Xtrain.append(feat)
            self.ytrain.append(label)

        self.Xtrain = np.stack(self.Xtrain, axis=0)


    def cluster(self):
        '''Cluster keystroke information'''
        print("Learning Clusters...")
        clustering = KMeans(n_clusters=50).fit(self.Xtrain)
        return clustering.labels_

    def hmm(self, clusters):
        '''Use HMM's to learn keystroke information'''
        print("Learning Hidden Markov Model...")
        # Learn transition probabilities from tv corpora
        transmat = self.getTransitionProb()

        model = hmm.GaussianHMM(n_components=27, covariance_type="diag", n_iter=1000)
        model.transmat_ = transmat
        model.fit(clusters)
        return model


    def getTransitionProb(self):
        '''Calculate transition probabilties from txtsrc'''
        print("Get Transition Probabilities...")
        t_prob_file = "out/raw_sentence/transitionprob.txt"
        if os.path.exists(t_prob_file):
            return np.loadtxt(t_prob_file)
        src_files = []
        for path, dirs, files in os.walk('txtsrc'):
            for file in files:
                src_files.append(os.path.join(path, file))

        t_prob = np.zeros((27, 27))

        for file in src_files:
            with open(file, "r") as f:
                data = f.read().lower()
                data = " ".join(re.findall("[a-z]+", data))
                for i in range(1, len(data)):
                    t_prob[self.convertletter(data[i-1])][self.convertletter(data[i])] += 1

        t_prob = sklearn.preprocessing.normalize(t_prob, axis=1, norm='l1')
        np.savetxt(t_prob_file, t_prob)
        return t_prob

    def convertletter(self, l):
        if l == " ":
            return 26
        return ord(l) - ord('a')

    def convertnumber(self, n):
        if n == 26:
            return " "
        return chr(n + ord('a'))

    def print_predict(self, output):
        string = ""
        for o in output:
            string += self.convertnumber(o)
        return string

    def accuracy(self, model, prediction):
        total = len(self.ytrain)
        correct = 0
        for i, p in enumerate(prediction):
            if self.ytrain[i] == p:
                correct += 1

        return correct/total



def main():
    '''Run learn keystrokes'''
    files = sys.argv[1:]
    outfile = os.path.join("out", "raw_sentence", str(datetime.datetime.now()) + "_raw")
    print("--- Classify Keystrokes ---")

    classifier = ClassifyKeystrokes(files)



main()
