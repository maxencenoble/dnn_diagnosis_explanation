import h5py
import numpy as np

LIST_OF_ALL_ECG = ["DI", "DII", "DIII", "AVL", "AVF", "AVR", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_ecg(tracings_file="./data/ecg_tracings.hdf5",
            patient_id=0,
            type_ecg="all"):
    """Returns the array of ECG from patient_id
    :param
        - type_ecg= all or D or A or V"""
    assert(patient_id < 827, 'id of patient too high')
    with h5py.File(tracings_file, "r") as f:
        table_ecg = np.array(f['tracings'])
        ecg_patient = table_ecg[patient_id]
        ecg=[]
        if type_ecg == "all":
            list_of_ecg = np.arange(12)
        elif type_ecg == "D":
            list_of_ecg = np.arange(3)
        elif type_ecg == "A":
            list_of_ecg = np.arange(3, 6)
        elif type_ecg == "V":
            list_of_ecg = np.arange(6, 12)
        else:
            list_of_ecg = [LIST_OF_ALL_ECG.index(type_ecg)]
        for k in list_of_ecg:
            ecg.append(ecg_patient[:, k])
    return np.array(ecg)


if __name__ == "__main__":
    all_ecgs = get_ecg()
    print(all_ecgs)
