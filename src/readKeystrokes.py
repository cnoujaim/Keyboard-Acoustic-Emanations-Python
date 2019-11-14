"""
Written by Shoyo Inokuchi (March 2019)
Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""

import os
import sys

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



# File input (single WAV file path -> sound data encoded as array)

def wav_read(filepath):
    """Return 1D NumPy array of wave-formatted audio data denoted by filename.
    Input should be a string containing the path to a wave-formatted audio file.
    """
    sample_rate, data = wav.read(filepath)
    if type(data[0]) == np.ndarray:
        return data[:, 0]
    else:
        return data


# Sound preprocessing before keystroke detection

def silence_threshold(sound_data, n=5, factor=11, output=True):
    """Return the silence threshold of the sound data.
    The sound data should begin with n-seconds of silence.
    """
    sampling_rate = 44100
    num_samples   = sampling_rate * n
    silence       = sound_data[:num_samples]
    print(np.amax(silence))

    return np.amax(silence)

def detect_keystrokes(sound_data, sample_rate=44100, output=True):
    """Return slices of sound_data that denote each keystroke present.

    Objective:
    - Satisfy same functional requirements as 'detect_keystrokes()', but better
    - Create a more accurate and flexible keystroke detection function
      utilizing more advanced audio processing techniques
    - Calculate MFCC etc. of sound_data to detect relevant peaks in sound
    """
    threshold          = silence_threshold(sound_data, output=output)
    keystroke_duration = 0.05  # seconds
    len_sample         = int(sample_rate * keystroke_duration)
    keystrokes = []

    peaks, properties = scipy.signal.find_peaks(sound_data,threshold=517, distance=len_sample)
    print(f"Found {len(peaks)} keystrokes in data")

    for i in peaks:
        a, b = i, i + len_sample
        if b > len(sound_data):
            b = len(sound_data)

        keystroke = sound_data[a:b]
        keystrokes.append(keystroke)

    return np.array(keystrokes)


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
    for i in range(n):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(f'Index: {i}')
        plt.plot(keystrokes[i])
    plt.show()


def main():
    filepath = str(sys.argv[1])
    outfile = os.path.join("out", "keystrokes", filepath.split("/")[-1] + "_out")
    wav_data = wav_read(filepath)

    keystrokes = detect_keystrokes(wav_data)
    np.savetxt(outfile, keystrokes)
    visualize_keystrokes(filepath)

main()
