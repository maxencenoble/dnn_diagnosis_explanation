import h5py
import argparse
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('classic')
# %matplotlib inline


sns.set()

parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
parser.add_argument('--tracings', default="./ecg_tracings.hdf5",  # or date_order.hdf5
                    help='HDF5 containing ecg tracings.')
parser.add_argument('--patient', default="0", help='patient id')

args, unk = parser.parse_known_args()
if unk:
    warnings.warn("Unknown arguments:" + str(unk) + ".")

with h5py.File(args.tracings, "r") as f:
    x = np.array(f['tracings'])
    print(np.shape(x))
    ecg_patient = x[int(args.patient)]
    abscisse = np.arange(0, 4096)
    for k in range(12):
        ecg = ecg_patient[:, k]
        plt.plot(abscisse, ecg)
    plt.show()
