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
parser.add_argument('--ecg', default="all", help='which curve to plot')

args, unk = parser.parse_known_args()

assert (args.patient < 827, 'id of patient too high')

list_of_all_ecg = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]

if unk:
    warnings.warn("Unknown arguments:" + str(unk) + ".")

with h5py.File(args.tracings, "r") as f:
    table_ecg = np.array(f['tracings'])
    ecg_patient = table_ecg[int(args.patient)]
    time_scale = np.arange(0, 4096)

    if args.ecg == "all":
        list_of_ecg = np.arange(12)
    elif args.ecg == "D":
        list_of_ecg = np.arange(3)
    elif args.ecg == "A":
        list_of_ecg = np.arange(3, 6)
    elif args.ecg == "V":
        list_of_ecg = np.arange(6, 12)
    else:
        list_of_ecg = [list_of_all_ecg.index(args.ecg)]
    for k in list_of_ecg:
        ecg = ecg_patient[:, k]
        plt.plot(time_scale, ecg, label=list_of_all_ecg[k])
    plt.show()
