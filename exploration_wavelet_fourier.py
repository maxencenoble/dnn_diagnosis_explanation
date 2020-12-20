import sys
import h5py
import pywt
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as sdec

import utils.illustration as uil
import utils_ecg.read_ecg as read_ecg

LEN = 4096
NB_PATIENT = 827
TRACINGS_FILE = "./dnn_files/data/ecg_tracings.hdf5"
CMAP = plt.get_cmap("Set1")
DISEASE_LIST = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
DISEASE_DIC = {0: "Non malade",
               1: "1dAVb",
               2: "RBBB",
               3: "LBBB",
               4: "SB",
               5: "AF",
               6: "ST"
               }

DIC_SIZE_WAVELET = {0: 140,
                    1: 140,
                    2: 268,
                    3: 523,
                    4: 1033,
                    5: 2054}

ANNOTATIONS_CSV = np.genfromtxt("./dnn_files/data/annotations/dnn.csv", delimiter=',')[1:, 1:]
ANNOTATIONS_CSV_pd = pd.read_csv("./dnn_files/data/annotations/dnn.csv", delimiter=',')

# reading all the file
with h5py.File(TRACINGS_FILE, "r") as f:
    TABLE_ECG = np.array(f['tracings'])

"""Below are useful functions to extract wavelet coefficients from ECG data or to rebuild the signal"""


def get_coeff_multi_level(ecg, sigma_threshold=1., type_threshold='hard', family='sym7', level=5):
    """Returns the list of coefficients from the decomposition of the ECG obtained with the parameters.
    (list of arrays)"""
    coeffs = pywt.wavedec(ecg, family, level=level)
    thresh = sigma_threshold * np.sqrt(2 * np.log(LEN))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], thresh, type_threshold)
    return coeffs


def reconstruct_signal(coeffsth, family='sym7'):
    """Returns the ECG obtained with a list of coefficients with same size as get_coeff_multi_level(...,family=family,...)
     (array) """
    return pywt.waverec(coeffsth, family)


def error(real_ecg, rec_ecg):
    """Returns the quadratic error between two ECG"""
    return np.mean((real_ecg - rec_ecg) * (real_ecg - rec_ecg))


def get_indices_list_above_threshold(type_ecg=0,
                                     sigma_threshold=1.,
                                     threshold_coeff=0.01,
                                     type_threshold='hard',
                                     family='sym7',
                                     level=5):
    """ Function used on all the ECGS.

    Returns the indexes of the coefficients obtained from decomposition with parameters where:
        - MEAN(ABS(coeff))_{all the ECGs} >threshold

    Shape : list of arrays (coeffs)
            list of arrays (corresponding names) :
                - e : for type of ecg (0,...11)
                - l : for level (coefficient wavelet)
                - c : for index (coefficient wavelet)
                - sc : for index (coefficient scaling)
    """
    all_coeffs = np.array(
        [get_coeff_multi_level(TABLE_ECG[patient_id, :, type_ecg], sigma_threshold, type_threshold, family,
                               level=level) for patient_id
         in range(NB_PATIENT)])
    liste_indices = []
    liste_names = ["e" + str(type_ecg) + "_sc" + str(i) for i in range(DIC_SIZE_WAVELET[0])]
    for i in range(1, level + 1):
        abs_mean = np.abs(np.mean(all_coeffs[:, i]))
        selected = np.where(abs_mean >= threshold_coeff)
        liste_indices.append(selected)
        liste_names += ["e" + str(type_ecg) + "_l" + str(level - i + 1) + "_c" + str(indice) for indice in selected[0]]
    return liste_indices, liste_names


def from_array_coeffs_to_coeffs_with_zeros(array_coeffs, liste_indices_non_zero):
    """ Returns the list of wavelet coeffs with threshold applied :
    Parameters:
        - array_coeffs = np.array(140 + size of non zero coeffs after threshold)
        - liste_indices_non_zero = obtained with get_indices_list_above_threshold
    Shape : list of arrays
    """
    coeffs_wavelet = []
    coeffs_wavelet.append(array_coeffs[:DIC_SIZE_WAVELET[0]])
    array_coeffs = array_coeffs[DIC_SIZE_WAVELET[0]:]
    for i in range(1, 6):
        array_coeffs_i = np.zeros(DIC_SIZE_WAVELET[i])
        array_coeffs_i[liste_indices_non_zero[i - 1]] = array_coeffs[:len(liste_indices_non_zero[i - 1][0])]
        array_coeffs = array_coeffs[len(liste_indices_non_zero[i - 1][0]):]
        coeffs_wavelet.append(array_coeffs_i)
    return coeffs_wavelet


def from_ecg_to_array_coeffs(ecg,
                             liste_indices_non_zero,
                             sigma_threshold=1.,
                             type_threshold='hard',
                             family='sym7',
                             level=5):
    """ Returns the array of coefficients which finally represent the ECG
    Parameter:
        - liste_indices_non_zero : obtained with get_indices_list_above_threshold[0]
    Shape : array"""
    coeffs = get_coeff_multi_level(ecg, sigma_threshold, type_threshold, family, level=level)
    all_coeffs = [coeffs[0]]
    for i in range(1, 6):
        all_coeffs.append(coeffs[i][liste_indices_non_zero[i - 1]])
    return np.concatenate(all_coeffs)


"""How to get final coefficient data from ECG data ? -> turn_ecg_into_coeffs """


def turn_ecg_into_coeffs(type_disease,
                         threshold_coeff=0.03,
                         sigma_threshold=1.,
                         type_threshold='hard',
                         family='sym7',
                         level=5
                         ):
    """Returns the data from ECG through a transformation into coefficients

    Parameters:
        - type_disease (str) : 1dAVb, RBBB,...

    Shape : dictionary
        - data : array (827 x nb_coeffs_selected per type ECG with concatenation of the 12 ECG)
        - target : array 827 (label from type_disease annotation)
        - feature_names : array (names of the coefficients, len : sum nb_coeffs_selected all type ECG)
        - target_names : array (matching with target, len : 2)
    """

    liste_all_names = []
    liste_coeffs_features = [[] for i in range(NB_PATIENT)]
    list_nb_coeffs = []
    list_non_zero = []
    for type_ecg in range(12):
        liste_indices_non_zero_0, liste_names = get_indices_list_above_threshold(type_ecg=type_ecg,
                                                                                 threshold_coeff=threshold_coeff,
                                                                                 sigma_threshold=sigma_threshold,
                                                                                 type_threshold=type_threshold,
                                                                                 family=family,
                                                                                 level=level)
        liste_all_coeff = np.array([from_ecg_to_array_coeffs(TABLE_ECG[id, :, type_ecg], liste_indices_non_zero_0,
                                                             sigma_threshold=sigma_threshold,
                                                             type_threshold=type_threshold,
                                                             family=family,
                                                             level=level
                                                             ) for id in range(NB_PATIENT)])
        list_nb_coeffs.append(np.shape(liste_all_coeff)[1])
        list_non_zero.append(liste_indices_non_zero_0)

        liste_coeffs_features = np.concatenate((liste_coeffs_features, liste_all_coeff), axis=1)

        liste_all_names = np.concatenate((liste_all_names, liste_names))

    res = {'data': liste_coeffs_features,
           'target': ANNOTATIONS_CSV_pd[type_disease].values,
           'feature_names': liste_all_names,
           'target_names': np.array(['sain', 'malade']),
           'nb_coeffs_per_type_ecg': list_nb_coeffs,
           'liste_indices_non_zero_0_per_type_ecg': list_non_zero}
    return res


def turn_coeffs_into_ecg(res_coeffs, family='sym7'):
    """Returns the data from coefficients through a transformation into ECG

    Parameters:
        - res_coeffs : dictionary with same shape as turn_ecg_into_coeffs(... , family=family,...)

    Returns :
    array (827,4096,12)

    """
    list_nb_coeffs = res_coeffs['nb_coeffs_per_type_ecg']
    all_coeffs = res_coeffs['data']
    list_non_zero = res_coeffs['liste_indices_non_zero_0_per_type_ecg']
    res_signal = []
    for id in range(np.shape(all_coeffs)[0]):
        signal_patient = []
        coeffs = all_coeffs[id]
        for j in range(12):
            signal_patient.append(
                reconstruct_signal(from_array_coeffs_to_coeffs_with_zeros(coeffs[:list_nb_coeffs[j]], list_non_zero[j]),
                                   family=family))
        signal_patient = np.array(signal_patient).T
        res_signal.append(signal_patient)
    return np.array(res_signal)


"""EXAMPLE OF USE"""

# res_coeffs = turn_ecg_into_coeffs(DISEASE_DIC[1], threshold_coeff=0.005)
# signal = turn_coeffs_into_ecg(res_coeffs)
# id_patient=30
# coeffs = res_coeffs["data"][0]
# print(coeffs[coeffs>0])
# for j in range(12):
# plt.plot(TABLE_ECG[id_patient, :, j], label="true")
# plt.plot(signal[id_patient, :, j], label="new")
# plt.legend()
# plt.show()

"""PCA towards wavelet case"""


def pca_visualisation_wavelet(id_ecg=0,
                              family='db5',
                              level=5,
                              sigma=1.0,
                              type_threshold='hard'):
    """
    Parameters :
        - id_ecg : int (0,...,11)
        - classic parameters with wavelet decomposition

    Plots multiple curves:
        - distribution of quantity of non zero coeffs (rate and number)
        - PCA : analysis of variance
        - PCA : (PC1,PC2) et (PC1,PC3) with label
    """

    ECG = TABLE_ECG[:, :, id_ecg]
    print("taille de la donnée ECG : ", np.shape(ECG))

    # wavelet
    coeffs_brut = pywt.wavedec(ECG, family, level=level)
    print("nb de famille de coefficients :", len(coeffs_brut))
    thresh = sigma * np.sqrt(2 * np.log(LEN))
    for j in tqdm(range(NB_PATIENT)):
        for i in range(1, len(coeffs_brut)):
            coeffs_brut[i][j] = pywt.threshold(coeffs_brut[i][j], thresh, type_threshold)
    ECG_db = np.concatenate(coeffs_brut, axis=1)
    print("dimension feature coeffs wavelet : ", len(ECG_db[0]))

    non_zero_coeffs = np.sum(np.abs(ECG_db) > 0.0001, axis=1)

    plt.hist(non_zero_coeffs, bins=100, color='green')
    plt.title("Distribution du nb de coeffs non nuls pour les features wavelet")
    plt.show()

    plt.hist(non_zero_coeffs / (len(ECG_db[0])), bins=100, color='red')
    plt.title("Distribution du remplissage non nul des features wavelet")
    plt.show()

    # PCA
    pca = sdec.PCA()
    ECG_db_pca = pca.fit_transform(ECG_db)

    # explainibility of variance
    fig = plt.figure(figsize=(15, 10))
    uil.plot_variance_acp(fig, pca, ECG_db_pca)
    plt.show()

    # matching with labels
    COLOR_DIC = {k: CMAP(k - 1) if k != 0 else CMAP(11) for k, v in DISEASE_DIC.items()}
    count = 0
    colors = [COLOR_DIC[0]] * NB_PATIENT
    for disease in DISEASE_DIC.values():
        if disease != "Non malade":
            count += 1
            for i in range(NB_PATIENT):
                if colors[i] == COLOR_DIC[0]:
                    colors[i] = COLOR_DIC[ANNOTATIONS_CSV_pd[disease][i] * count]
    # les couleurs se chevauchent dans l'ordre des annotations

    markersizes = [20] * NB_PATIENT

    fig = plt.figure(figsize=(15, 15))

    # definition of the axes
    for nbc, nbc2, count in [(1, 2, 1), (1, 3, 2)]:
        ax = fig.add_subplot(1, 2, count)
        uil.plot_pca(ax, ECG_db_pca, pca, nbc, nbc2, colors, markersizes)

    # Build legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_DIC[act], marker=".", linestyle=None, markersize=25, label=DISEASE_DIC[act]) for
        act in range(0, 7)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0, 1), fontsize=10)
    plt.show()


"""PCA towards fft case"""


def pca_visualisation_fft(id_ecg=0, sigma=1.0):
    """
    Parameters :
        - id_ecg : int (0,...,11)

    Plots multiple curves:
        - distribution of quantity of non zero coeffs (rate and number)
        - PCA : analysis of variance
        - PCA : (PC1,PC2) et (PC1,PC3) with label
    """
    ECG = TABLE_ECG[:, :, id_ecg]
    print("taille de la donnée ECG : ", np.shape(ECG))

    # preprocessing
    nouv_ECG = np.array([read_ecg.symetrize_bounds(ecg) for ecg in ECG])

    # fourier
    fourier = np.fft.fft(nouv_ECG, axis=1)
    threshold = sigma * np.sqrt(2 * np.log(LEN))
    for freq_list in fourier:
        freq_list[np.abs(freq_list) < threshold] = 0

    print("dimension feature coeffs fft : ", len(fourier[0]))

    non_zero_coeffs = np.sum(np.abs(fourier) > 0.0001, axis=1)

    plt.hist(non_zero_coeffs, bins=100, color='green')
    plt.title("Distribution du nb de coeffs non nuls pour les features fft")
    plt.show()

    plt.hist(non_zero_coeffs / (len(fourier[0])), bins=100, color='red')
    plt.title("Distribution du remplissage non nul des features fft")
    plt.show()

    # PCA
    pca = sdec.PCA()
    fourier_db_pca = pca.fit_transform(np.abs(fourier))

    # explainibility of variance
    fig = plt.figure(figsize=(15, 10))
    uil.plot_variance_acp(fig, pca, fourier_db_pca)
    plt.show()

    # matching with labels
    COLOR_DIC = {k: CMAP(k - 1) if k != 0 else CMAP(11) for k, v in DISEASE_DIC.items()}
    count = 0
    colors = [COLOR_DIC[0]] * NB_PATIENT
    for disease in DISEASE_DIC.values():
        if disease != "Non malade":
            count += 1
            for i in range(NB_PATIENT):
                if colors[i] == COLOR_DIC[0]:
                    colors[i] = COLOR_DIC[ANNOTATIONS_CSV_pd[disease][i] * count]
    # les couleurs se chevauchent dans l'ordre des annotations

    markersizes = [20] * NB_PATIENT

    fig = plt.figure(figsize=(15, 15))

    # definition of the axes
    for nbc, nbc2, count in [(1, 2, 1), (1, 3, 2)]:
        ax = fig.add_subplot(1, 2, count)
        uil.plot_pca(ax, fourier_db_pca, pca, nbc, nbc2, colors, markersizes)

    # Build legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_DIC[act], marker=".", linestyle=None, markersize=25, label=DISEASE_DIC[act]) for
        act in range(0, 7)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0, 1), fontsize=10)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print('Parameters : type_decomposition id_ecg sigma [family level type_threshold]')
    else:
        type_decomposition = sys.argv[1]
        id_ecg = int(sys.argv[2])
        sigma = float(sys.argv[3])
        if type_decomposition == 'fft':
            pca_visualisation_fft(id_ecg, sigma)
        else:
            family = sys.argv[4]
            level = int(sys.argv[5])
            type_threshold = sys.argv[6]
            pca_visualisation_wavelet(id_ecg, family, level, sigma, type_threshold)
