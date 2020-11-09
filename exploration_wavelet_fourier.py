import sys
import h5py
import pywt
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as sdec

import utils.illustration as uil
import read_ecg

LEN = 4096
NB_PATIENT = 827
TRACINGS_FILE = "./data/ecg_tracings.hdf5"
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

ANNOTATIONS_CSV = np.genfromtxt("./data/annotations/dnn.csv", delimiter=',')[1:, 1:]
ANNOTATIONS_CSV_pd = pd.read_csv("./data/annotations/dnn.csv", delimiter=',')

# lecture de tout le fichier
with h5py.File(TRACINGS_FILE, "r") as f:
    TABLE_ECG = np.array(f['tracings'])


def pca_visualisation_wavelet(id_ecg=0,
                              family='db5',
                              level=5,
                              sigma=1.0,
                              type_threshold='hard'):
    """Affiche plusieurs graphes:
        - distribution du nb de coeffs non nuls (brut et taux)
        - PCA : analyse de la variance
        - PCA : (PC1,PC2) et (PC1,PC3) avec label"""
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


def pca_visualisation_fft(id_ecg=0, sigma=1.0):
    """Affiche plusieurs graphes:
        - distribution du nb de coeffs non nuls (brut et taux)
        - PCA : analyse de la variance
        - PCA : (PC1,PC2) et (PC1,PC3) avec label"""
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
