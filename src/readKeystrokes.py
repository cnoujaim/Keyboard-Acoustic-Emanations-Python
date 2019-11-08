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
    tolerance     = 85.52671939449968
    measured      = np.std(silence)
    if output and measured > tolerance:
        # raise Exception(f'Sound data must begin with at least {n}s of silence.')
        print(f'Initial silence was higher than expected at {measured}, which',
              f' is higher than expected {tolerance}')
    return max(np.amax(silence), abs(np.amin(silence))) * factor


def remove_random_noise(sound_data, threshold=None):
    """Return a copy of sound_data where random noise is replaced with 0's.
    The original sound_data is not mutated.
    """
    threshold = threshold or silence_threshold(sound_data)
    sound_data_copy = deepcopy(sound_data)
    for i in range(len(sound_data_copy)):
        if abs(sound_data_copy[i]) < threshold:
            sound_data_copy[i] = 0
    return sound_data_copy


# Keystroke detection (encoded array -> all keystroke data in array)

def detect_keystrokes(sound_data, sample_rate=44100, output=True):
    """Return slices of sound_data that denote each keystroke present.
    Returned keystrokes are coerced to be the same length by appending trailing
    zeros.
    Current algorithm:
    - Calculate the "silence threshold" of sound_data.
    - Traverse sound_data until silence threshold is exceeded.
    - Once threshold is exceeded, mark that index as "a".
    - Identify the index 0.3s ahead of "a", and mark that index as "b".
    - If "b" happens to either:
          1) denote a value that value exceeds the silence threshold (aka:
             likely impeded on the next keystroke waveform)
          2) exceed the length of sound_data
      then backtrack "b" until either:
          1) it denotes a value lower than the threshold
          2) "b" is 1 greater than "a"
    - Slice sound_data from index "a" to "b", and append that slice to the list
      to be returned. If "b" was backtracked, then pad the slice with trailing
      zeros to make it 0.3s long.
    :type sound_file  -- NumPy array denoting input sound clip
    :type sample_rate -- integer denoting sample rate (samples per second)
    :rtype            -- NumPy array of NumPy arrays
    """
    print("------- DETECT KEYSTROKE --------")
    threshold          = silence_threshold(sound_data, output=output)
    keystroke_duration = 0.05   # seconds
    len_sample         = int(sample_rate * keystroke_duration)

    keystrokes = []
    i = 0
    while i < len(sound_data):
        if abs(sound_data[i]) > threshold:
            a, b = i, i + len_sample
            if b > len(sound_data):
                b = len(sound_data)
            while abs(sound_data[b]) > threshold and b > a:
                b -= 1
            keystroke = sound_data[a:b]
            trailing_zeros = np.array([0 for _ in range(len_sample - (b - a))])
            keystroke = np.concatenate((keystroke, trailing_zeros))
            keystrokes.append(keystroke)
            i = b - 1
        i += 1
    return np.array(keystrokes)


def detect_keystrokes_improved(sound_data, sample_rate=44100):
    """Return slices of sound_data that denote each keystroke present.

    Objective:
    - Satisfy same functional requirements as 'detect_keystrokes()', but better
    - Create a more accurate and flexible keystroke detection function
      utilizing more advanced audio processing techniques
    - Calculate MFCC etc. of sound_data to detect relevant peaks in sound
    """
    pass


# Display detected keystrokes (WAV file -> all keystroke graphs)

def visualize_keystrokes(filepath):
    print("------- VISUALIZE KEYSTROKES --------")
    """Display each keystroke detected in WAV file specified by filepath."""
    wav_data = wav_read(filepath)
    keystrokes = detect_keystrokes(wav_data)
    n = len(keystrokes)
    print(f'Number of keystrokes detected in "{filepath}": {n}')
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
    #visualize_keystrokes("recordings/testing6_phone_noise_reduced.wav")
    wav_data = wav_read(filepath)
    keystrokes = detect_keystrokes(wav_data)
    np.savetxt(outfile, keystrokes)

main()
