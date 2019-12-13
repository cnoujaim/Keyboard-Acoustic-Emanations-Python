"""
Written by Shoyo Inokuchi (March 2019)
Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""

import os
import sys
import json
import math

from copy import deepcopy

import numpy as np
import wavio
import matplotlib.pyplot as plt
import sqlalchemy as db
import sqlalchemy.orm as orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql

from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange


import scipy
import scipy.signal

SAMPLE_RATE = 44100


def plotSpectrum(y,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[:2000]
    print(len(frq))

    Y = fft(y)/n # fft computing and normalization
    Y = Y[:2000]

    return frq, abs(Y)

def find_peaks(data, distance=1, threshold=0.1):
    peaks = []
    i = 0
    while i < len(data):
        max_loc = np.argmax(data[i:i+distance]) + i
        # print(max_loc)
        # print(data[max_loc])
        if data[max_loc] > threshold:
            peaks.append(max_loc)
            i = max_loc + distance
        else:
            i += 1000

    return peaks



# File input (single WAV file path -> sound data encoded as array)

def wav_read(filepath):
    """Return 1D NumPy array of wave-formatted audio data denoted by filename.
    Input should be a string containing the path to a wave-formatted audio file.
    """
    data = wavio.read(filepath)
    data = data.data
    if type(data[0]) == np.ndarray:
        data = data[:,0]
    data = data.astype(np.float32)
    data = (data / np.max(np.abs(data)))
    data -= np.mean(data)
    return data


def detect_keystrokes(sound_data, sample_rate=SAMPLE_RATE, output=True, num_peaks = None, labels = None):
    """Return slices of sound_data that denote each keystroke present.

    Objective:
    - Satisfy same functional requirements as 'detect_keystrokes()', but better
    - Create a more accurate and flexible keystroke detection function
      utilizing more advanced audio processing techniques
    - Calculate MFCC etc. of sound_data to detect relevant peaks in sound
    """
    keystroke_duration = 0.2  # seconds
    len_sample         = int(sample_rate * keystroke_duration)
    keystrokes = []

    peaks = find_peaks(sound_data, threshold=0.06, distance=len_sample)
    print(f"Found {len(peaks)} keystrokes in data")

    if not num_peaks:
        labels = [None for i in range(len(peaks))]

    for i, p in enumerate(peaks):
        p = p - 1440
        a, b = p, p + int(0.2 * sample_rate)
        if b > len(sound_data):
            b = len(sound_data)

        # print(p/SAMPLE_RATE)

        keystroke = sound_data[a:b]
        keystrokes.append((keystroke.tolist(), labels[i]))

    return keystrokes


# Display detected keystrokes (WAV file -> all keystroke graphs)

def visualize_keystrokes(filepath, label='a'):
    print("------- VISUALIZE KEYSTROKES --------")
    """Display each keystroke detected in WAV file specified by filepath."""
    wav_data = wav_read(filepath)
    keystrokes = detect_keystrokes(wav_data)
    n = 1
    print('Drawing keystrokes...')
    num_cols = 3
    num_rows = n/num_cols + 1
    # plt.figure(figsize=(num_cols * 6, num_rows * .75))
    for i in range(min(n, len(keystrokes))):

        plt.plot(np.array(keystrokes[i][0]))
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig(f"acoustic_signal_{keystrokes[i][1]}.png")
        plt.show()
        plt.close()

        frq, y = plotSpectrum(np.array(keystrokes[i][0]), SAMPLE_RATE)
        plt.plot(frq,y,'r') # plotting the spectrum
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.savefig(f"frequency_spectrum_{keystrokes[i][1]}.png")
        plt.show()
        plt.close()
    plt.show()


def getLabels(txtpath):
    if txtpath:
        with open(txtpath, 'r') as file:
            data = file.read().strip('\n')
        return len(data), [char for char in data]
    else:
        return None, None



def main():
    filepath = str(sys.argv[1])
    txtpath = None
    if len(sys.argv) > 2:
        txtpath = str(sys.argv[2])
    outfile = os.path.join("out", "keystrokes", filepath.split("/")[-1] + "_out")

    wav_data = wav_read(filepath)
    # f, t, Sxx = scipy.signal.spectrogram(wav_data, fs=SAMPLE_RATE)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    x_axis = [(i/SAMPLE_RATE) for i in range(len(wav_data))]
    plt.plot(x_axis, wav_data)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    title = filepath.split("/")[-1]
    plt.show()
    plt.savefig(f"fullfig_{title}.png")
    plt.close()


    num_peaks, labels = getLabels(txtpath)

    keystrokes = detect_keystrokes(wav_data, num_peaks = num_peaks, labels = labels)
    print(f"Keystrokes are the same length as labels is...: {len(keystrokes) == len(labels)}")
    with open(outfile, 'w') as f:
        f.write(json.dumps(keystrokes))
    visualize_keystrokes(filepath, label = "a")

main()
