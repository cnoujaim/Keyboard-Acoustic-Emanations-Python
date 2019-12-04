"""
Written by Shoyo Inokuchi (March 2019)
Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""

import os
import sys
import json

from copy import deepcopy

import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import sqlalchemy as db
import sqlalchemy.orm as orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql


import scipy
import scipy.signal

SAMPLE_RATE = 48000
BITS = 16.0


def find_peaks(data, distance=1, threshold=3000):
    peaks = []
    i = 0
    while i < len(data):
        max_loc = np.argmax(data[i:i+distance]) + i
        # print(max_loc / SAMPLE_RATE)
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
    sample_rate, data = wav.read(filepath)
    SAMPLE_RATE = sample_rate
    if type(data[0]) == np.ndarray:
        data = data[:, 0]
        data = data.astype(np.float32)
        data = (data / np.max(np.abs(data)))
        data -= np.mean(data)
        return  data #np.array(b)
    else:
        data = data.astype(np.float32)
        data = (data / np.max(np.abs(data)))
        data -= np.mean(data)
        return  data #np.array(b)



def detect_keystrokes(sound_data, sample_rate=SAMPLE_RATE, output=True, num_peaks = None, labels = None):
    """Return slices of sound_data that denote each keystroke present.

    Objective:
    - Satisfy same functional requirements as 'detect_keystrokes()', but better
    - Create a more accurate and flexible keystroke detection function
      utilizing more advanced audio processing techniques
    - Calculate MFCC etc. of sound_data to detect relevant peaks in sound
    """
    keystroke_duration = 0.5  # seconds
    len_sample         = int(sample_rate * keystroke_duration)
    keystrokes = []

    peaks = find_peaks(sound_data, threshold=3000, distance=len_sample)
    print(f"Found {len(peaks)} keystrokes in data")

    if not num_peaks:
        labels = [None for i in range(len(peaks))]

    for i, p in enumerate(peaks):
        p = p - 1440
        a, b = p, p + int(0.1 * sample_rate)
        if b > len(sound_data):
            b = len(sound_data)

        print(p / sample_rate)
        keystroke = sound_data[a:b]
        keystrokes.append((keystroke.tolist(), labels[i]))

    return keystrokes


# Display detected keystrokes (WAV file -> all keystroke graphs)

def visualize_keystrokes(filepath):
    print("------- VISUALIZE KEYSTROKES --------")
    """Display each keystroke detected in WAV file specified by filepath."""
    wav_data = wav_read(filepath)
    keystrokes = detect_keystrokes(wav_data)
    n = 20
    print('Drawing keystrokes...')
    num_cols = 3
    num_rows = n/num_cols + 1
    plt.figure(figsize=(num_cols * 6, num_rows * .75))
    for i in range(min(n, len(keystrokes))):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(f'Index: {i}')
        f, t, Sxx = scipy.signal.spectrogram(np.array(keystrokes[i][0]))
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        # plt.plot(np.array(keystrokes[i][0]))
        # plt.plot(np.array(keystrokes[i + 5][0]))
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
    f, t, Sxx = scipy.signal.spectrogram(wav_data, fs=SAMPLE_RATE)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    x_axis = [(i/SAMPLE_RATE) for i in range(len(wav_data))]
    plt.plot(x_axis, wav_data)
    plt.show()

    num_peaks, labels = getLabels(txtpath)

    keystrokes = detect_keystrokes(wav_data, num_peaks = num_peaks, labels = labels)
    with open(outfile, 'w') as f:
        f.write(json.dumps(keystrokes))
    visualize_keystrokes(filepath)

main()
