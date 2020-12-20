import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from dtw import accelerated_dtw
import scipy.signal

"""This file is for implementing all feature functions"""

"""Mono-features : functions which take a single tracing as an np.array and return a single float"""


def average(list_ecg):
    return np.mean(list_ecg)


def standard_deviation(list_ecg):
    return np.std(list_ecg)


def median(list_ecg):
    return np.median(list_ecg)


def median_absolute_value(list_ecg):
    return stats.median_abs_deviation(list_ecg)


def maximum(list_ecg):
    return np.amax(list_ecg)


def minimum(list_ecg):
    return np.amin(list_ecg)


def signal_magnitude_area(list_ecg):
    return np.sum(list_ecg)


def range(ecg):
    return np.amax(ecg) - np.amin(ecg)


def mid_range(ecg):
    return (np.amax(ecg) - np.amin(ecg)) / 2


def energy(list_ecg):
    return np.sum(list_ecg * list_ecg) / len(list_ecg)


def midhinge(list_ecg):
    q25, q75 = np.percentile(list_ecg, [25, 75])
    return (q25 + q75) / 2


def trimean(list_ecg):
    q25, q50, q75 = np.percentile(list_ecg, [25, 50, 75])
    return (q25 + 2 * q50 + q75) / 4


def interpercentile_range(x1=25, x2=75, normalize=False):
    """returns a lambda function which calculates the gap between percentile x2 and percentile x1 of the ecg,
    divided by the range of the ecg on normalize=True"""
    if normalize:
        return lambda ecg: (np.percentile(ecg, [x2])[0] - np.percentile(ecg, [x1])[0]) / (np.amax(ecg) - np.amin(ecg))
    else:
        return lambda ecg: np.percentile(ecg, [x2])[0] - np.percentile(ecg, [x1])[0]


def gm_asymetry(ecg):
    """Returns the highest value, the supremum"""
    percentiles = np.arange(start=60, stop=100, step=3)
    values = []
    for u in percentiles:
        qu, q50, q1_u = np.percentile(ecg, [u, 50, (100 - u)])
        values.append((qu + q1_u - 2 * q50) / (qu - q1_u))
    return max(values)


def gm_asymetry2(ecg):
    """Returns the most extreme value, supremum or infimum"""
    percentiles = np.arange(start=60, stop=100, step=3)
    values = []
    for u in percentiles:
        qu, q50, q1_u = np.percentile(ecg, [u, 50, (100 - u)])
        values.append((qu + q1_u - 2 * q50) / (qu - q1_u))
    mini = np.min(values)
    maxi = np.min(values)
    if np.abs(mini) > np.abs(maxi):
        return mini
    else:
        return maxi


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
    return int(60 * np.abs(freq[np.argmax(np.abs(fft))]))  # the number of steps


def correlation(x, y, tau):
    xx = np.append(x[tau:], np.zeros(tau))
    return np.sum(xx * np.conjugate(y))


def auto_correlation(ecg):
    """Computes the auto_correlation of an ecg lead with the lag corresponding to the heartbeat frequency"""
    lag = periods_via_fft(ecg)
    return correlation(ecg, ecg, lag) / np.sum(ecg * ecg)


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
    """
    Parameters :
        - list_ecg : (4096,12) corresponding
        - ecg_comparaison : ECG to compare considering asynchrony

    If plot :
        plots meaningful curves about asynchrony

    Returns: array (2,)
        0 : average all over the 12 ECGs of the l1 difference of indexes obtained with dtw
        1 : average all over the 12 ECGs of the l2 difference of indexes obtained with dtw
    """

    d2 = ecg_comparaison.reshape(-1, 1)
    res_l1 = 0
    res_l2 = 0
    for k in range(12):
        d1 = list_ecg[k].reshape(-1, 1)
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1, d2, dist='euclidean')
        res_l1 += np.sum((path[1] - path[0]) / 400)
        res_l2 += np.sum((path[1] - path[0]) * (path[1] - path[0]) / 16000)
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
    return np.array([res_l1 / 12, res_l2 / 12])
