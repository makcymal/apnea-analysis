import os
from shutil import move

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import firwin
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA


############
# EEG data #
############

# possible extensions of record files
# we need only REC
exts1 = ["ascii", "cm1", "cp1", "ch3", "cc1", "cn1", "cc3", "cu4"]
exts2 = ["cn4", "cn3", "cc2", "cs1", "REC", "txt"]


# rename and move EEG records for convinient access
def rename_dataset():
    for ext in exts1 + exts2:
        os.makedirs(f"dataset/{ext}", exist_ok=True)
        for pat in range(1, 41):
            for rec in range(1, 3):
                for ch in range(1, 3):
                    try:
                        move(
                            f"dataset/dataset_O{ch}/Np {pat}/Nr {rec}/N-{rec}.{ext}",
                            f"dataset/{ext}/{pat}-{rec}-O{ch}.{ext}",
                        )
                    except: # extension doesn't match
                        pass
                    

# get signal in form of amplitudes over fixed interval of time  
def read_signal(pat: int, rec: int, ch: int) -> list[int]:
    assert 1 <= pat <= 40
    assert 1 <= rec <= 2
    assert 1 <= ch <= 2

    with open(f"dataset/ascii/{pat}-{rec}-O{ch}.ascii", "r") as signal_file:
        signal = []
        for line in signal_file:
            signal.append(int(line))

    return signal


# plot signal, provide freq for correct time ticks
def visualize_signal(signal, freq=1):
    fig, ax = plt.subplots()
    x = np.linspace(0, len(signal) - 1, len(signal)) / freq
    _ = ax.plot(x, signal)


# build Finite Impulse Response filter coefficients
# pass only signal of frequency within [lo, hi] range
def fir_filter_coef(ntaps, lo, hi, freq) -> np.ndarray:
    coef = firwin(
        ntaps, [lo, hi], fs=freq, pass_zero=False, window="hamming", scale=False
    )
    return coef


# apply filter with coefficients on signal
# len of signal decreses by len(coef) - 1
def fir_filter(signal, coef) -> np.ndarray:
    return np.convolve(signal, coef, "valid")

# rhythms:
# delta: x <= 4 Hz
# theta: 4 Hz < x <= 8 Hz
# alpha: 8 Hz < x <= 13 Hz
# beta: 13 Hz < x <= 32 Hz
# gamma: 32 Hz < x (and <= 100 Hz for convinience)
GAMMA_TICKS = [4, 8, 13, 32, 100]
# healthy sleep assumes frequencies up to 40 Hz
# optionally we can ignore gamma rhythm
BETA_TICKS = [4, 8, 13, 100]


# integral power of each rhythm
def get_rhythms(signal: list[int], include_gamma=True) -> list[int]:
    # doing Fourier transform
    # going from time domain to frequency domain
    # y axis (amplitude)
    ampl = np.abs(rfft(signal))
    # x axis (frequency)
    freq = rfftfreq(len(signal), 1 / 200)

    # sum of amplitudes for each rhythm interval
    accum = [0]
    # total sum
    total = 0
    
    # current rhythm
    ticks = GAMMA_TICKS if include_gamma else BETA_TICKS
    curr_tick = 0

    # counting accum and total
    for a, f in zip(ampl, freq):
        if f > ticks[curr_tick]:
            accum.append(0)
            curr_tick += 1
        accum[-1] += a
        total += a

    # normalizing accum by dividing by total
    for i in range(len(accum)):
        accum[i] /= total
    # now we have percentage of power each rhythm has in the whole signal
    return accum


# converting rhythms into matrix
# each row - single patient and single record
# each row concatenates rhythms from both O1 and O2 channels
def get_rhythms_matrix(filt_coef):
    rhythms = []
    for pat in range(1, 41):
        for rec in range(1, 3):
            rhythms.append(
                get_rhythms(fir_filter(read_signal(pat, rec, 1), filt_coef))
                + get_rhythms(fir_filter(read_signal(pat, rec, 2), filt_coef))
            )
