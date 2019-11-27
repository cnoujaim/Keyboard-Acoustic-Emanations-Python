
import sys
import os
import datetime
import re
import sklearn

import numpy as np
from sklearn.cluster import KMeans
from librosa.feature import mfcc

from hmmlearn import hmm

class ClassifyKeystrokes:
    '''Classifies keystrokes using clustering and HMMs'''
    def __init__(self, files):
        self.keystrokes = []
        for file in files:
            self.keystrokes.append(np.loadtxt(file))
        self.keystrokes = np.vstack(self.keystrokes)
        self.Xtrain = []
        self.convert_keystrokes_to_features()
        self.hmm()


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
        for keystroke in self.keystrokes:
            feat = self.extract_features(keystroke)
            self.Xtrain.append(feat)

        self.Xtrain = np.stack(self.Xtrain, axis=0)
        # np.swapaxes(self.Xtrain, 0, 1)


    def cluster(self):
        '''Cluster keystroke information'''
        print("Learning Clusters...")
        clustering = KMeans(n_clusters=30, random_state=0).fit(self.Xtrain)
        print(clustering.labels_)

    def hmm(self):
        '''Use HMM's to learn keystroke information'''
        print("Learning Hidden Markov Model...")
        model = hmm.GaussianHMM(n_components=27, covariance_type="diag", n_iter=1000, init_params="st")
        # Learn transition probabilities from tv corpora
        transmat = self.getTransitionProb(model)
        for i in range(20):
            print(f"Run iteration {i}")
            # model.transmat_ = transmat
            model.fit(self.Xtrain)
            print(f"Log probabilitiy score is {model.score(self.Xtrain)}")
        Z2 = model.predict(self.Xtrain)
        print(Z2)

    def getTransitionProb(self, model):
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


def main():
    '''Run learn keystrokes'''
    files = sys.argv[1:]
    outfile = os.path.join("out", "raw_sentence", str(datetime.datetime.now()) + "_raw")
    print("--- Classify Keystrokes ---")

    classifier = ClassifyKeystrokes(files)



main()
