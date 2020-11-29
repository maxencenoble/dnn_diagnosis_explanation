import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from dtw import accelerated_dtw
import pandas as pd

import get_ecg as ge
import read_ecg as re

"""
This file is for implementing all feature functions.
"""

"""
Mono-features : functions which take a single tracing as an np.array and return a single float :

Median Absolute Value (X) = median(|X-E(X)|), more robust to outliers than std
Signal magnitude area = sum for a discrete signal
Energy = average sum of the squares
Interquartile range = the difference between 75th and 25th percentiles, aka between the upper and lower quartiles
Entropy = Shannon entropy of the value histogram of the signal = sum_p -p log(p)
Spectral entropy ? = Shannon entropy of the value histogram of the frequency spectrum
Auto_correlation : correlation between the signal and itself offsetted by 'lag'
Skewness : Skewness is the standardized moment of order three. For normally distributed data, the skewness should be about zero. A skewness value greater than zero means that there is more weight in the right tail of the distribution.
Kurtosis : Kurtosis is the standardized moment of order four.
"""


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


def auto_correlation(list_ecg, lag=10):
    return np.correlate(list_ecg, list_ecg, "same")[lag]


def auto_correlation_function(lag):
    return lambda list_ecg: np.abs(auto_correlation(list_ecg, lag))


def kurtosis(ecg):
    return stats.kurtosis(ecg)


def skewness(ecg):
    return stats.skew(ecg)


"""Poly-features : functions which take all tracing as an (12,_) np.array and return a single float

Average mean
Average standard deviation
Average 
"""


def average_mean(list_ecg):
    return np.mean(np.array([np.mean(list_ecg[k]) for k in range(12)]))


def average_std(list_ecg):
    return np.mean(np.array([np.std(list_ecg[k]) for k in range(12)]))


def average_kurtosis(list_ecg):
    return np.mean(np.array([stats.kurtosis(list_ecg[k]) for k in range(12)]))


def average_skewness(list_ecg):
    return np.mean(np.array([stats.skew(list_ecg[k]) for k in range(12)]))


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
