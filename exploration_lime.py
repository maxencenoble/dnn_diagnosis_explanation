from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import exploration_wavelet_fourier as wav
from keras.models import load_model
from keras.optimizers import Adam
import h5py
import argparse
import matplotlib.pyplot as plt

np.random.seed(1)

LEN = 4096
NB_PATIENT = 827
TRACINGS_FILE = "./data/ecg_tracings.hdf5"
PREDICTED_PROBA_FILE = np.load("./outputs/dnn_output.npy")
ANNOTATIONS_CSV_pd = pd.read_csv("./data/annotations/dnn.csv", delimiter=',')
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

parser = argparse.ArgumentParser(description='Get HTMLs recording LIME info')
parser.add_argument('--mode', default="explain", help='train for explanation otherwise checking')
parser.add_argument('--id_disease', default=1, help='Between 1 and 6')
parser.add_argument('--threshold_coeff', default=0., help='Select coeffs')
parser.add_argument('--id_patient', default=0, help='Select patient')
# --coeff
#   either :  e..._sc... (ECG - SCALINGCOEFF)
#   or     :  e..._l..._c... (ECG - LEVEL - COEFF)
args, unk = parser.parse_known_args()

##############TO MODIFY################################
LIST_TO_CHECK_COEFF = []
LIST_TO_CHECK_NEW_VALUE = []
#######################################################

THRESHOLD_COEFF = float(args.threshold_coeff)
ID_DISEASE = int(args.id_disease)
ID_PATIENT_TO_EXPLAIN = int(args.id_patient)
MAX_PROBA_DISEASE = np.max(PREDICTED_PROBA_FILE[:, ID_DISEASE - 1])

with h5py.File(TRACINGS_FILE, "r") as f:
    TABLE_ECG = np.array(f['tracings'])


# plt.plot(TABLE_ECG[ID_PATIENT_TO_EXPLAIN, :, 6], label="true")


def list_id_patient_with_disease(id_disease):
    return np.where(ANNOTATIONS_CSV_pd[DISEASE_DIC[id_disease]].values == 1)


# print("malades")
# print(PREDICTED_PROBA_FILE[list_id_patient_with_disease(ID_DISEASE), ID_DISEASE - 1])

print(" Création du fichier des coeffs ECG: ...")
ECG_COEFFS = wav.turn_ecg_into_coeffs(type_disease=DISEASE_DIC[ID_DISEASE],
                                      threshold_coeff=THRESHOLD_COEFF)
print(" Création du fichier des coeffs ECG: OK")
ecg_signals = wav.turn_coeffs_into_ecg(ECG_COEFFS)

print("Chargement du modèle : ...")
model = load_model("./model/model.hdf5", compile=False)
model.compile(loss='binary_crossentropy', optimizer=Adam())
print("Chargement du modèle : OK")

print("Split : ...")
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(ECG_COEFFS["data"],
                                                                                  ECG_COEFFS["target"],
                                                                                  train_size=0.80)
print("Split : OK")

print("Explainer tabulaire : ...")
explainer = lime.lime_tabular.LimeTabularExplainer(train,
                                                   feature_names=ECG_COEFFS["feature_names"],
                                                   class_names=ECG_COEFFS["target_names"],
                                                   discretize_continuous=True)
print("Explainer tabulaire : OK")


def feature_index(x):
    """unique index de feature_names correspondant à x"""
    return np.where(ECG_COEFFS["feature_names"] == x)[0][0]


def predict_proba(x):
    """x : np.array (nb samples x nb features)"""
    # turn ecg coeffs into ecg signals
    new_x = {'data': x,
             'nb_coeffs_per_type_ecg': ECG_COEFFS['nb_coeffs_per_type_ecg'],
             'liste_indices_non_zero_0_per_type_ecg': ECG_COEFFS['liste_indices_non_zero_0_per_type_ecg']
             }
    # plt.plot(wav.turn_coeffs_into_ecg(new_x)[0, :, 6])
    y_score = model.predict(wav.turn_coeffs_into_ecg(new_x), batch_size=32, verbose=1)
    y_score = y_score[:, ID_DISEASE - 1]
    # y_score = y_score * MAX_PROBA_DISEASE / 0.16
    inverted_proba = 1. - y_score
    return np.array([inverted_proba, y_score]).T


new = predict_proba(ECG_COEFFS["data"])[:, 1]
former = model.predict(TABLE_ECG, batch_size=32, verbose=1)[:, ID_DISEASE - 1]

plt.plot(former, label="former")
plt.plot(new, label="new")
plt.legend()
plt.show()


def explainer_patient(id_patient=0, nb_features=25):
    exp = explainer.explain_instance(ECG_COEFFS["data"][id_patient],
                                     predict_proba,
                                     num_features=nb_features,
                                     top_labels=1)
    exp.save_to_file("./output_lime/disease_" + DISEASE_DIC[ID_DISEASE] + "/patient_" + str(id_patient) + ".html")


if args.mode == "explain":
    id_patient_liste = list_id_patient_with_disease(ID_DISEASE)[0]

    print("All patients with disease :", id_patient_liste)
    for id_patient in id_patient_liste:
        print("Explainer autour du patient " + str(id_patient) + "...")
        explainer_patient(id_patient)
        print("Explainer autour du patient " + str(id_patient) + ": OK")
else:
    long_check = len(LIST_TO_CHECK_COEFF)
    for j in range(long_check):
        print("Checking coeff " + LIST_TO_CHECK_COEFF[j] + " on patient " + str(ID_PATIENT_TO_EXPLAIN))
        temp = ECG_COEFFS["data"][ID_PATIENT_TO_EXPLAIN].copy()
        print("Value before :", temp[feature_index(LIST_TO_CHECK_COEFF[j])])
        print("P(disease) before :", predict_proba(np.array([temp]))[0, 1])
        temp[feature_index(LIST_TO_CHECK_COEFF[j])] = LIST_TO_CHECK_NEW_VALUE[j]
        # temp=np.zeros(len(temp))
        print("Value after :", temp[feature_index(LIST_TO_CHECK_COEFF[j])])
        print("P(disease) after :", predict_proba(np.array([temp]))[0, 1])
        plt.show()
        print()
