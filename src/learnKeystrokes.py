
import sys
import os

import numpy as np
from sklearn.cluster import SpectralClustering
from librosa.feature import mfcc

class ClassifyKeystrokes:
    '''Classifies keystrokes using clustering and HMMs'''
    def __init__(self, infile):
        self.keystrokes = np.loadtxt(infile)
        self.Xtrain = []
        self.convert_keystrokes_to_features()
        self.cluster()


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

    def cluster(self):
        '''Cluster keystroke information'''
        clustering = SpectralClustering(n_clusters=30, assign_labels="discretize", random_state=0).fit(self.Xtrain)
        print(clustering.labels_)


def main():
    infile = str(sys.argv[1])
    outfile = os.path.join("out", "raw_sentence", infile.split("/")[-1] + "_raw")

    classifier = ClassifyKeystrokes(infile)



    with open(outfile, "w") as f:
        f.write("ehllo my namm is crissina")


main()
