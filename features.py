import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

import get_ecg as ge
import read_ecg as re

"""
This file is for implementing all feature functions.

The expected data type of function parameters are numpy arrays of shape (12,_)
which should be called on tracings where the zeros have already been removed.

These functions will all either return :
- a numpy array of shape (12) containing the results of a function on all 12 leads of the ecg 
- a single float (like average correlation between signals)
"""

"""Average"""
def feature_avg(list_ecg):
    return np.mean(list_ecg, axis=1)


"""Standard Deviation"""
def feature_std(list_ecg):
    return np.std(list_ecg, axis=1)


"""Median Absolute Value (X) = median(|X-E(X)|), more robust to outliers than std"""
def feature_mad(list_ecg):
    return stats.median_abs_deviation(list_ecg, axis=1)


""" Maximum and minimum"""
def feature_max(list_ecg):
    return np.amax(list_ecg, axis=1)

def feature_min(list_ecg):
    return np.amin(list_ecg, axis=1)


""" Signal magnitude area = sum for a discrete signal"""
def feature_sma(list_ecg):
    return np.sum(list_ecg, axis=1)


""" Energy = average sum of the squares"""
def feature_energy(list_ecg):
    return np.sum(list_ecg * list_ecg, axis=1)


""" Interquartile range = the difference between 75th and 25th percentiles, aka between the upper and lower quartiles"""
def feature_iqr(list_ecg):
    q75, q25 = np.percentile(list_ecg, [75, 25], axis=1)
    return q75 - q25


""" Entropy = Shannon entropy of the value histogram of the signal = sum_p -p log(p)"""
def feature_entropy(list_ecg, numbins=100, base=None):
    """ numbins : number of bins to use for the histogram
    base : base of the log for entropy calculation """
    return np.array([stats.entropy(stats.relfreq(list_ecg[k], numbins).frequency, None, base) for k in range(np.shape(list_ecg)[0])])


# Spectral entropy ? = Shannon entropy of the value histogram of the frequency spectrum

""" Auto_correlation : correlation between the signal and itself offsetted by 'lag'"""
def feature_auto_correlation(list_ecg, lag):
    return np.array(np.correlate(list_ecg[k], list_ecg[k], "same")[lag] for k in range(np.shape(list_ecg)[0]))


""" TRASH / TESTS

temp = ge.get_ecg()
ecg_example = np.array([re.delete_zeros(temp[k]) for k in range(12)])

avg = feature_avg(ecg_example)
std = feature_std(ecg_example)
mad = feature_mad(ecg_example)
maxi = feature_max(ecg_example)
mini = feature_min(ecg_example)
sma = feature_sma(ecg_example)
energy = feature_energy(ecg_example)
iqr = feature_iqr(ecg_example)
entropy = feature_entropy(ecg_example)

print('Shape of example ecg : '+str(np.shape(ecg_example)))
test = np.array([[6,7,8,9],[0,1,2,3],[-1,0,0,1]])

print('Energy of test = '+str(feature_energy(test)))
print('Energy of ecg_example = '+str(feature_energy(ecg_example)))


print('IQR of test = '+str(feature_iqr(test)))
print('IQR of ecg_example = '+str(feature_iqr(ecg_example)))


print('Entropy of test = ' + str(feature_entropy(test,base=2)))
print('Entropy of ecg example = ' + str(feature_entropy(ecg_example,numbins=1000,base=2)))


histo = stats.relfreq(ecg_example[0],numbins=100).frequency
#plt.plot(histo,label='Histogram of ecg tracing')
#plt.legend()
#plt.show()
print('entropy='+str(stats.entropy(histo)))


print('Autocorrelation of test for lag = 1 : ' + str(feature_auto_correlation(test,1)))
print('Autocorrelation of ecg example for lag =1 : ' + str(feature_auto_correlation(ecg_example,1)))



X = np.arange(10,5000,step=50)
Y = np.array([feature_entropy(ecg_example,numbins=k,base=2) for k in X])
#plt.plot(X,Y)
#plt.title('Variation of histogram entropy with respect to the number of bins used for the histogram')
#plt.legend()
#plt.show()


X = np.arange(np.shape(ecg_example)[1])
Y = np.correlate(ecg_example[0],ecg_example[0],"same")
#plt.plot(X,Y)
#plt.title('Auto-correlation of the first lead signal')
#plt.show()
"""