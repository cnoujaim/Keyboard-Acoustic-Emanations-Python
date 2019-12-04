import sys
import os
import datetime
import re
import sklearn
import json
import copy

from collections import defaultdict
import numpy as np
from sklearn.cluster import SpectralClustering
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

    def train(self, iters=1):
        print("Training model...")
        best_model = None
        best_clus = None
        best_acc = 0
        for i in range(iters):
            print(f"-- Run iteration {i} --")
            clusters = self.cluster()
            clusters = np.reshape(clusters, (-1, 1))
            model = self.hmm(clusters)
            print(f"Log probability for this model is {model.score(clusters)}")
            # acc = self.accuracy(model, model.predict(clusters))
            # print(f"Accuracy for this model is {acc}")
            # if acc > best_acc:
            #     best_acc = acc
            #     best_model = copy.deepcopy(model)
            #     best_clus = copy.deepcopy(clusters)
            print(f"Prediction for this model is {self.print_predict(model.predict(clusters))}")

        # for c, p, a in zip(best_clus, best_model.predict(best_clus), self.ytrain):
        #     print(f"cluster {c}, prediction {self.convertnumber(p)}, actual {self.convertnumber(a)}")

        # print(f"Prediction for this model is {self.print_predict(best_model.predict(best_clus))}")


    def extract_features(self, keystroke, sr=48000, n_mfcc=32, hop_len=120):
        '''Return an MFCC-based feature vector for a given keystroke.'''
        spec = mfcc(y=keystroke.astype(float),
                    sr=sr,
                    n_mfcc=n_mfcc,
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
        print(self.Xtrain.shape)


    def cluster(self):
        '''Cluster keystroke information'''
        print("Learning Clusters...")
        clustering = SpectralClustering(n_clusters=2).fit(self.Xtrain)
        print(clustering.labels_)
        return clustering.labels_

    def hmm(self, clusters):
        '''Use HMM's to learn keystroke information'''
        print("Learning Hidden Markov Model...")
        # Learn transition probabilities from tv corpora
        transmat = self.getTransitionProb()

        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, init_params="mcst")
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
        if l is None:
            return None
        if ord('a') > ord(l) or ord('z') < ord(l):
            return 26
        return ord(l) - ord('a')

    def convertnumber(self, n):
        if n is None:
            return None
        if n == 26:
            return " "
        return chr(n + ord('a'))

    def print_predict(self, output):
        string = ""
        for o in output:
            string += self.convertnumber(o)
        return string

    def accuracy(self, model, prediction):
        key_acc = {}
        total = len(self.ytrain)
        correct = 0
        for i, p in enumerate(prediction):
            if self.ytrain[i] not in key_acc:
                key_acc[self.ytrain[i]] = [0, 0]
            key_acc[self.ytrain[i]][1] += 1
            if self.ytrain[i] == p:
                correct += 1
                key_acc[self.ytrain[i]][0] += 1


        for key in key_acc:
            val = key_acc[key]
            print(f"key {self.convertnumber(key)} and acc {val[0]/val[1]} with {val[1]} total")
        return correct/total



def main():
    '''Run learn keystrokes'''
    files = sys.argv[1:]
    outfile = os.path.join("out", "raw_sentence", str(datetime.datetime.now()) + "_raw")
    print("--- Classify Keystrokes ---")

    classifier = ClassifyKeystrokes(files)



main()
