import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from dtw import accelerated_dtw
import scipy.signal
import pandas as pd

import get_ecg as ge
import read_ecg as re

"""This file is for implementing all feature functions"""


"""Mono-features : functions which take a single tracing as an np.array and return a single float"""


def average(list_ecg):
    return np.mean(list_ecg)


def standard_deviation(list_ecg):
    return np.std(list_ecg)


def median_absolute_value(list_ecg):
    return stats.median_abs_deviation(list_ecg)


def maximum(list_ecg):
    return np.amax(list_ecg)


def minimum(list_ecg):
    return np.amin(list_ecg)


def signal_magnitude_area(list_ecg):
    return np.sum(list_ecg)


def energy(list_ecg):
    return np.sum(list_ecg * list_ecg) / len(list_ecg)


def interquartile_range(list_ecg):
    q75, q25 = np.percentile(list_ecg, [75, 25])
    return q75 - q25


def entropy(list_ecg, numbins=100, base=None):
    """ numbins : number of bins to use for the histogram
    base : base of the log for entropy calculation """
    return stats.entropy(stats.relfreq(list_ecg, numbins).frequency, None, base)


def kurtosis(ecg):
    return stats.kurtosis(ecg)


def skewness(ecg):
    return stats.skew(ecg)


def periods_via_fft(ecg):
    """Takes a single lead ecg and returns the most probable heartbeat period in number of steps (1 step = 1/400 s)"""
    """Compute the Butterworth filter"""
    butter_a, butter_b = scipy.signal.butter(N=2, Wn=[50 / 60, 120 / 60], btype='bandpass', analog=False, output='ba',
                                             fs=400)
    filtered_ecg = scipy.signal.lfilter(butter_a, butter_b, ecg)
    signal_length = np.size(ecg)
    fft = np.fft.fft(filtered_ecg)
    freq = np.fft.fftfreq(signal_length, 1 / 400)
    return int(400 / np.abs(freq[np.argmax(np.abs(fft))]))  # the number of steps


def frequencies_via_fft(ecg):
    """Takes a single lead ecg and returns the most probable heartbeat frequency in bpm (1 step = 1/400 s)"""
    """Compute the Butterworth filter"""
    butter_a, butter_b = scipy.signal.butter(N=2, Wn=[50 / 60, 120 / 60], btype='bandpass', analog=False, output='ba',
                                             fs=400)
    filtered_ecg = scipy.signal.lfilter(butter_a, butter_b, ecg)
    signal_length = np.size(ecg)
    fft = np.fft.fft(filtered_ecg)
    freq = np.fft.fftfreq(signal_length, 1 / 400)
    return int(60*np.abs(freq[np.argmax(np.abs(fft))]))  # the number of steps

def correlation(x,y,tau):
    xx = np.append(x[tau:],np.zeros(tau))
    return np.sum(xx*np.conjugate(y))

def auto_correlation(ecg):
    """Computes the auto_correlation of an ecg lead with the lag corresponding to the heartbeat frequency"""
    lag = periods_via_fft(ecg)
    return correlation(ecg,ecg,lag)

"""Poly-features : functions which take all tracing as an (12,_) np.array and return a single float"""


def average_mean(list_ecg):
    return np.mean(np.array([np.mean(list_ecg[k]) for k in range(12)]))


def average_std(list_ecg):
    return np.mean(np.array([np.std(list_ecg[k]) for k in range(12)]))


def average_kurtosis(list_ecg):
    return np.mean(np.array([stats.kurtosis(list_ecg[k]) for k in range(12)]))


def average_skewness(list_ecg):
    return np.mean(np.array([stats.skew(list_ecg[k]) for k in range(12)]))


def period_via_fft(ecg_list):
    """Takes a 12-lead ecg and returns the most probable heartbeat period in number of steps (1 step = 1/400 s)
    To do so it takes the frequency of maximum amplitude in the Fourier spectrum of each lead after passing it through
    a Butterworth filter of order 2 of bandpass [50 bpm, 120 bpm], then takes the frequency which appears the most
    across all leads"""
    potential_periods = []
    butter_a, butter_b = scipy.signal.butter(N=2, Wn=[50 / 60, 120 / 60], btype='bandpass', analog=False, output='ba',
                                             fs=400)
    for k in range(12):
        filtered_ecg = scipy.signal.lfilter(butter_a, butter_b, ecg_list[k])
        signal_length = np.size(ecg_list[k])
        fft_k = np.fft.fft(filtered_ecg)
        freq_k = np.fft.fftfreq(signal_length, 1 / 400)
        resu = int(400 / np.abs(freq_k[np.argmax(np.abs(fft_k))]))
        """converts frequency in Hz -> period in number of steps"""
        potential_periods.append(resu)
    return max(potential_periods, key=potential_periods.count)


def frequency_via_fft(ecg_list):
    """Takes a 12-lead ecg and returns the most probable heartbeat frequency in bpm (1 step = 1/400 s)
    To do so it takes the frequency of maximum amplitude in the Fourier spectrum of each lead after passing it through
    a Butterworth filter of order 2 of bandpass [50 bpm, 120 bpm], then takes the frequency which appears the most
    across all leads"""
    potential_frequencies = []
    butter_a, butter_b = scipy.signal.butter(N=2, Wn=[50 / 60, 120 / 60], btype='bandpass', analog=False, output='ba',
                                             fs=400)
    for k in range(12):
        filtered_ecg = scipy.signal.lfilter(butter_a, butter_b, ecg_list[k])
        signal_length = np.size(ecg_list[k])
        fft_k = np.fft.fft(filtered_ecg)
        freq_k = np.fft.fftfreq(signal_length, 1 / 400)
        resu = int(60 * np.abs(freq_k[np.argmax(np.abs(fft_k))]))
        """converts frequency in Hz -> frequency in bpm"""
        potential_frequencies.append(resu)
    return max(potential_frequencies, key=potential_frequencies.count)


def average_asynchrony(list_ecg, ecg_comparaison, plot=False):
    """Compared to some ecg_comparaison"""

    d2 = ecg_comparaison.reshape(-1, 1)
    res = 0
    for k in range(12):
        d1 = list_ecg[k].reshape(-1, 1)
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1, d2, dist='euclidean')
        res += np.sum((path[1] - path[0]) / 16000)
        if plot:
            figure = plt.figure(figsize=(10, 10))
            ax1 = figure.add_subplot(2, 1, 1)
            ax1.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            ax1.plot(path[0], path[1], 'w')
            ax1.set_xlabel('ECG 0')
            ax1.set_ylabel('ECG 1')
            ax1.set_title(f'DTW Minimum Path with minimum distance: {np.round(d, 2)}')

            ax2 = figure.add_subplot(2, 1, 2)
            ax2.plot(list_ecg[0], label="ecg0")
            ax2.plot(list_ecg[1], label="ecg1")
            ax2.set_title("Comparaison des ECG")
            ax2.legend()

            plt.show()
    return res / 12
