import h5py
import numpy as np
import matplotlib.pyplot as plt

"""Pre processing of an ECG"""


def delete_zeros(ecg):
    """Delete the first and the last occurences of zero in the ecg"""
    temp = ecg[np.where(np.abs(np.array(ecg)) > 10 ** (-6))[0][0]:]
    return temp[:np.where(np.abs(np.array(temp)) > 10 ** (-6))[-1][-1] + 1]


def symetrize(ecg):
    """Symetrizes the ecg by adding the reversed signal at the end"""
    return np.concatenate((ecg, np.flip(ecg)))


def symetrize_bounds(ecg):
    return np.append(ecg, ecg[0])


"""Fourier transformation of an ECG"""


def fft(ecg, symetric="bounds", nb_frequency=10):
    """
    Plots : signal, fft, rebuilt signal with threshold on fft
    Returns :
        - peaks : the nb_frequency most important frequencies of the signal regarding abs(fft)
        - value_peaks : corresponding abs(fft) values
    """
    if symetric != "bounds":
        signal = symetrize(delete_zeros(ecg))
    else:
        signal = symetrize_bounds(delete_zeros(ecg))

    t = np.arange(len(signal))

    # plotting the signal=f(time)
    plt.subplot(311)
    plt.plot(t, signal)
    plt.ylabel("signal")

    # fft
    fourier = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))

    # plotting the fft=f(frequency) with threshold
    plt.subplot(312)
    plt.plot(freq, np.abs(fourier))
    threshold = np.sort(np.abs(fourier))[::-1][nb_frequency]
    print(threshold)
    plt.axhline(y=threshold, color='r', linestyle='-', label='threshold')
    plt.ylabel("module fft")

    # select the frequencies with the biggest module
    mask = np.abs(fourier) > threshold
    peaks = freq[mask]
    value_peaks = np.abs(fourier)[mask]

    # plotting the ifft=f(time) with regards to threshold values
    plt.subplot(313)
    fourier[np.abs(fourier) < threshold] = 0
    inv_fourier = np.fft.ifft(fourier)
    plt.plot(inv_fourier)
    plt.ylabel("rebuilt signal")
    plt.show()

    return peaks, value_peaks


if __name__ == "__main__":
    with h5py.File("../dnn_files/data/ecg_tracings.hdf5", "r") as f:
        table_ecg = np.array(f['tracings'])
        ecg_patient = table_ecg[0]
        print(delete_zeros(ecg_patient[:, 0]))
        print(symetrize(np.array([2, 3, 4])))
        peaks, values = fft(ecg_patient[:, 0])
        print("peaks")
        print(peaks)
        print("values")
        print(values)
        plt.subplot(211)
        plt.plot(delete_zeros(ecg_patient[:, 0]))
        plt.ylabel("signal")
        plt.subplot(212)
        # plt.plot(np.diff(ecg_patient[:, 0]))

        correlation = np.correlate(delete_zeros(ecg_patient[:, 0]), delete_zeros(ecg_patient[:, 0]), "same")
        print("correlation", correlation)
        plt.plot(correlation)
        plt.ylabel("cross_correlation")

        plt.show()
