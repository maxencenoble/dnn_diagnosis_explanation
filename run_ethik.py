import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import get_ecg as ge
import read_ecg as re
import ethik
import features as ft
from ethik.cache_explainer import CacheExplainer
from ethik.utils import set_fig_size, to_pandas

explainer = ethik.ClassificationExplainer()

""" Imports the DNN annotations as a dataframe """
dnn_annotations = pd.read_csv("./data/annotations/dnn.csv")
gold_standard = pd.read_csv("./data/annotations/gold_standard.csv")

""" Imports and cleans the raw signals """


def is_only_zero(li):
    return (np.sum(np.abs(li) < 0.0001) == len(li))


def has_dead_lead(lili):  # shape = (p,t)
    (p, t) = np.shape(lili)
    verdict = False
    for lead in range(p):
        verdict = verdict or is_only_zero(lili[lead])
    return verdict


def get_clean_ecg(tracings_file="./data/ecg_tracings.hdf5"):
    with h5py.File(tracings_file, "r") as f:
        ecg_tracings = np.swapaxes(np.array(f['tracings']), 1, 2)  # shape = (827,12,4096)
        bad_indexes = []
        for i in range(827):
            if has_dead_lead(ecg_tracings[i]):
                bad_indexes += [i]
        ecg_tracings = np.delete(ecg_tracings, bad_indexes, 0)  # Isolate and delete the bad indexes
    return (bad_indexes,
            np.array([[re.delete_zeros(ecg[lead]) for lead in range(12)] for ecg in ecg_tracings]))  # delete zeroes


(bad_indexes, table_ecg) = get_clean_ecg()
N_patients = np.shape(table_ecg)[0]
"""table_ecg has shape (N_patients, 12, _ (e.g. 4096))"""

print('*** Finished importing and cleaning the raw signals ***')

"""Removes the bad signals from the labels dataframes"""
dnn_annotations = dnn_annotations.drop(bad_indexes)
gold_standard = gold_standard.drop(bad_indexes)

print('*** Finished cleaning the label dataframes ***')

"""Names the different leads. These arrays have shape (813,_ (e.g. 4096))"""

table_ecg0 = table_ecg[:, 0]
table_ecg1 = table_ecg[:, 1]
table_ecg2 = table_ecg[:, 2]
table_ecg3 = table_ecg[:, 3]
table_ecg4 = table_ecg[:, 4]
table_ecg5 = table_ecg[:, 5]
table_ecg6 = table_ecg[:, 6]
table_ecg7 = table_ecg[:, 7]
table_ecg8 = table_ecg[:, 8]
table_ecg9 = table_ecg[:, 9]
table_ecg10 = table_ecg[:, 10]
table_ecg11 = table_ecg[:, 11]


def plot_mono_feature_influence_dnn(
        feat_function, patho_list=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], max_range=0.1,
        name_of_function='default', save=0
):
    """Plots or save the Ethik influence of feat_function on pathology prediction by the DNN for pathologies in patho_list for every lead in the ECG.
    One can choose the upper bound of the drawing with max_range, as well as the name of the function to display in the title with name_of_function.
    Plots on save=0, saves on save=1
    """

    """Computes the feature values matrix"""
    feature_values = np.empty((N_patients, 12))
    for id in range(N_patients):
        feature_values[id] = np.array([feat_function(table_ecg[id][lead]) for lead in range(12)])

    """extracts the name of the function"""
    function_name = str(feat_function)[10:]
    i = 0
    while function_name[i] != " ":
        i += 1
    function_name = function_name[:i]

    """Build the feature name list"""
    names = [function_name + "_of_lead_" + str(k) for k in range(12)]

    """Build the feature dataframe"""
    series = {
        names[0]: feature_values[:, 0],
        names[1]: feature_values[:, 1],
        names[2]: feature_values[:, 2],
        names[3]: feature_values[:, 3],
        names[4]: feature_values[:, 4],
        names[5]: feature_values[:, 5],
        names[6]: feature_values[:, 6],
        names[7]: feature_values[:, 7],
        names[8]: feature_values[:, 8],
        names[9]: feature_values[:, 9],
        names[10]: feature_values[:, 10],
        names[11]: feature_values[:, 11]
    }
    df_feature = pd.DataFrame(series)

    """Plot or save"""
    for pathology in patho_list:

        fig = explainer.plot_influence(
            X_test=df_feature[
                [names[0], names[1], names[2], names[3], names[4], names[5], names[6], names[7], names[8], names[9],
                 names[10], names[11]]],
            y_pred=dnn_annotations[pathology],
            yrange=[0, max_range])

        """Declare the name of the graph"""
        tit = ''
        if name_of_function == 'default':
            tit = 'Influence of ' + function_name + ' on ' + pathology + ' prediction by the DNN'
        else:
            tit = 'Influence of ' + name_of_function + ' on ' + pathology + ' prediction by the DNN'
        fig.update_layout(title=tit)

        if save:
            if name_of_function == 'default':
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + function_name + "_" + pathology + "_dnn.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + function_name + "_" + pathology + "_dnn as png in affichage_ecg/ethik/" + pathology + "/")
            else:
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + name_of_function + "_" + pathology + "_dnn.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + name_of_function + "_" + pathology + "_dnn as png in affichage_ecg/ethik/" + pathology + "/")



        else:
            fig.show()
    return 0


def plot_poly_feature_influence_dnn(
        feat_function, patho_list=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], max_range=0.1,
        name_of_function='default', save=0
):
    """Plots or save the Ethik influence of feat_function on pathology prediction by the DNN for pathologies in patho_list the 12-lead ECG.
    One can choose the upper bound of the drawing with max_range, as well as the name of the function to display in the title with name_of_function.
    Plots on save=0, saves on save=1
    """

    """Computes the feature values matrix"""
    feature_values = np.zeros(N_patients)
    for id in range(N_patients):
        feature_values[id] = feat_function(table_ecg[id])

    """extracts the name of the function"""
    function_name = str(feat_function)[10:]
    i = 0
    while function_name[i] != " ":
        i += 1
    function_name = function_name[:i]

    """Builds the feature dataframe"""
    series = {
        function_name: feature_values
    }
    df_feature = pd.DataFrame(series)

    """Plot or save"""
    for pathology in patho_list:
        fig = explainer.plot_influence(
            X_test=df_feature[function_name],
            y_pred=dnn_annotations[pathology],
            yrange=[0, max_range])

        """Declare the name of the graph"""
        tit = ''
        if name_of_function == 'default':
            tit = 'Influence of ' + function_name + ' on ' + pathology + ' prediction by the DNN'
        else:
            tit = 'Influence of ' + name_of_function + ' on ' + pathology + ' prediction by the DNN'
        fig.update_layout(title=tit)

        if save:
            if name_of_function == 'default':
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + function_name + "_" + pathology + "_dnn.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + function_name + "_" + pathology + "_dnn as png in affichage_ecg/ethik/" + pathology + "/")
            else:
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + name_of_function + "_" + pathology + "_dnn.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + name_of_function + "_" + pathology + "_dnn as png in affichage_ecg/ethik/" + pathology + "/")

        else:
            fig.show()
    return 0


def plot_mono_feature_influence_gold(
        feat_function, patho_list=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], max_range=0.1,
        name_of_function='default', save=0
):
    """Plots or save the Ethik influence of feat_function on pathology prediction by the Gold Standard for pathologies in patho_list for every lead in the ECG.
    One can choose the upper bound of the drawing with max_range, as well as the name of the function to display in the title with name_of_function.
    Plots on save=0, saves on save=1
    """

    """Computes the feature values matrix"""
    feature_values = np.empty((N_patients, 12))
    for id in range(N_patients):
        feature_values[id] = np.array([feat_function(table_ecg[id][lead]) for lead in range(12)])

    """extracts the name of the function"""
    function_name = str(feat_function)[10:]
    i = 0
    while function_name[i] != " ":
        i += 1
    function_name = function_name[:i]

    """Build the feature name list"""
    names = [function_name + "_of_lead_" + str(k) for k in range(12)]

    """Build the feature dataframe"""
    series = {
        names[0]: feature_values[:, 0],
        names[1]: feature_values[:, 1],
        names[2]: feature_values[:, 2],
        names[3]: feature_values[:, 3],
        names[4]: feature_values[:, 4],
        names[5]: feature_values[:, 5],
        names[6]: feature_values[:, 6],
        names[7]: feature_values[:, 7],
        names[8]: feature_values[:, 8],
        names[9]: feature_values[:, 9],
        names[10]: feature_values[:, 10],
        names[11]: feature_values[:, 11]
    }
    df_feature = pd.DataFrame(series)

    """Plot or save"""
    for pathology in patho_list:
        fig = explainer.plot_influence(
            X_test=df_feature[
                [names[0], names[1], names[2], names[3], names[4], names[5], names[6], names[7], names[8], names[9],
                 names[10], names[11]]],
            y_pred=gold_standard[pathology],
            yrange=[0, max_range])

        """Declare the name of the graph"""
        tit = ''
        if name_of_function == 'default':
            tit = 'Influence of ' + function_name + ' on ' + pathology + ' prediction by the Gold Standard'
        else:
            tit = 'Influence of ' + name_of_function + ' on ' + pathology + ' prediction by the Gold Standard'
        fig.update_layout(title=tit)

        if save:
            if name_of_function == 'default':
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + function_name + "_" + pathology + "_gold.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + function_name + "_" + pathology + "_gold as png in affichage_ecg/ethik/" + pathology + "/")
            else:
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + name_of_function + "_" + pathology + "_gold.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + name_of_function + "_" + pathology + "_gold as png in affichage_ecg/ethik/" + pathology + "/")
        else:
            fig.show()
    return 0


def plot_poly_feature_influence_gold(
        feat_function, patho_list=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'], max_range=0.1,
        name_of_function='default', save=0
):
    """Plots or save the Ethik influence of feat_function on pathology prediction by the Gold Standard for pathologies in patho_list the 12-lead ECG.
    One can choose the upper bound of the drawing with max_range, as well as the name of the function to display in the title with name_of_function.
    Plots on save=0, saves on save=1
    """

    """Computes the feature values matrix"""
    feature_values = np.zeros(N_patients)
    for id in range(N_patients):
        feature_values[id] = feat_function(table_ecg[id])

    """extracts the name of the function"""
    function_name = str(feat_function)[10:]
    i = 0
    while function_name[i] != " ":
        i += 1
    function_name = function_name[:i]

    """Builds the feature dataframe"""
    series = {
        function_name: feature_values
    }
    df_feature = pd.DataFrame(series)

    """Plot or save"""
    for pathology in patho_list:
        fig = explainer.plot_influence(
            X_test=df_feature[function_name],
            y_pred=gold_standard[pathology],
            yrange=[0, max_range])

        """Declare the name of the graph"""
        tit = ''
        if name_of_function == 'default':
            tit = 'Influence of ' + function_name + ' on ' + pathology + ' prediction by the Gold Standard'
        else:
            tit = 'Influence of ' + name_of_function + ' on ' + pathology + ' prediction by the Gold Standard'
        fig.update_layout(title=tit)

        if save:
            if name_of_function == 'default':
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + function_name + "_" + pathology + "_gold.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + function_name + "_" + pathology + "_gold as png in affichage_ecg/ethik/" + pathology + "/")
            else:
                fig.write_image(
                    "affichage_ecg/ethik/" + pathology + "/influence_" + name_of_function + "_" + pathology + "_gold.png",
                    width=1280, height=840)
                print(
                    "Correctly saved: influence_" + function_name + "_" + name_of_function + "_gold as png in affichage_ecg/ethik/" + pathology + "/")
        else:
            fig.show()
    return 0


mono_features = [ft.average, ft.standard_deviation, ft.median_absolute_value, ft.maximum, ft.minimum,
                 ft.signal_magnitude_area, ft.energy, ft.entropy,
                 ft.kurtosis, ft.skewness, ft.periods_via_fft, ft.frequencies_via_fft, ft.auto_correlation]
poly_features = [ft.average_mean, ft.average_std, ft.average_skewness, ft.average_kurtosis]

"""Saving all the tracings"""


def save_all_mono_feature_influences_dnn(mono_features_list=mono_features):
    for feat in mono_features_list:
        plot_mono_feature_influence_dnn(feat, save=1)
    return 0


def save_all_poly_feature_influences_dnn(poly_features_list=poly_features):
    for feat in poly_features_list:
        plot_poly_feature_influence_dnn(feat, save=1)
    return 0


def save_all_mono_feature_influences_gold(mono_features_list=mono_features):
    for feat in mono_features_list:
        plot_mono_feature_influence_gold(feat, save=1)
    return 0


def save_all_poly_feature_influences_gold(poly_features_list=poly_features):
    for feat in poly_features_list:
        plot_poly_feature_influence_gold(feat, save=1)
    return 0


# save_all_mono_feature_influences_dnn()
# save_all_poly_feature_influences_dnn()

# save_all_mono_feature_influences_gold()
# save_all_poly_feature_influences_gold()


"""Most recent functions (not yet printed) : """

new_mono_features = [ft.range, ft.mid_range, ft.median, ft.midhinge, ft.trimean, ft.gm_asymetry, ft.gm_asymetry2]

"""
save_all_mono_feature_influences_dnn(new_mono_features)
save_all_mono_feature_influences_gold(new_mono_features)


plot_mono_feature_influence_dnn(ft.interpercentile_range(10,90),name_of_function='interpercentile_range_10%_90%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(10,90),name_of_function='interpercentile_range_10%_90%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(25,75),name_of_function='interpercentile_range_25%_75%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(25,75),name_of_function='interpercentile_range_25%_75%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(40,60),name_of_function='interpercentile_range_40%_60%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(40,60),name_of_function='interpercentile_range_40%_60%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(0,50),name_of_function='interpercentile_range_0%_50%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(0,50),name_of_function='interpercentile_range_0%_50%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(50,100),name_of_function='interpercentile_range_50%_100%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(50,100),name_of_function='interpercentile_range_50%_100%',save=1)


plot_mono_feature_influence_dnn(ft.interpercentile_range(10,90,normalize=True),name_of_function='normalized_interpercentile_range_10%_90%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(10,90,normalize=True),name_of_function='normalized_interpercentile_range_10%_90%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(25,75,normalize=True),name_of_function='normalized_interpercentile_range_25%_75%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(25,75,normalize=True),name_of_function='normalized_interpercentile_range_25%_75%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(40,60,normalize=True),name_of_function='normalized_interpercentile_range_40%_60%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(40,60,normalize=True),name_of_function='normalized_interpercentile_range_40%_60%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(0,50,normalize=True),name_of_function='normalized_interpercentile_range_0%_50%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(0,50,normalize=True),name_of_function='normalized_interpercentile_range_0%_50%',save=1)

plot_mono_feature_influence_dnn(ft.interpercentile_range(50,100,normalize=True),name_of_function='normalized_interpercentile_range_50%_100%',save=1)
plot_mono_feature_influence_gold(ft.interpercentile_range(50,100,normalize=True),name_of_function='normalized_interpercentile_range_50%_100%',save=1)
"""


# plot_mono_feature_influence_dnn(ft.auto_correlation,save=1)
# plot_mono_feature_influence_gold(ft.auto_correlation,save=1)


"""Saving the influence rankings as Excel files"""


def average_asynchrony_0(list_ecg):
    return ft.average_asynchrony(list_ecg, ecg_comparaison=table_ecg[0, 0])


def explain_mono_feature_influence(pathology, mono_features_list=mono_features, tau_lim=0.7):
    feature_list = []
    type_ecg = []
    proba_mean_dnn = []
    proba_low_dnn = []
    proba_high_dnn = []
    proba_mean_gs = []
    proba_low_gs = []
    proba_high_gs = []

    for feat_function in mono_features_list:
        """Computes the feature values matrix"""
        feature_values = np.empty((N_patients, 12))
        for id in range(N_patients):
            feature_values[id] = np.array([feat_function(table_ecg[id][lead]) for lead in range(12)])

        """extracts the name of the function"""
        function_name = str(feat_function)[10:]
        i = 0
        while function_name[i] != " ":
            i += 1
        function_name = function_name[:i]

        """Build the feature name list"""
        names = ["lead_" + str(k) for k in range(12)]

        """Build the feature dataframe"""
        series = {
            names[0]: feature_values[:, 0],
            names[1]: feature_values[:, 1],
            names[2]: feature_values[:, 2],
            names[3]: feature_values[:, 3],
            names[4]: feature_values[:, 4],
            names[5]: feature_values[:, 5],
            names[6]: feature_values[:, 6],
            names[7]: feature_values[:, 7],
            names[8]: feature_values[:, 8],
            names[9]: feature_values[:, 9],
            names[10]: feature_values[:, 10],
            names[11]: feature_values[:, 11]
        }
        df_feature = pd.DataFrame(series)

        explanation_dnn = explainer.explain_influence(
            X_test=df_feature[
                [names[0], names[1], names[2], names[3], names[4], names[5], names[6], names[7], names[8], names[9],
                 names[10], names[11]]],
            y_pred=dnn_annotations[pathology])
        explanation_gold_standard = explainer.explain_influence(
            X_test=df_feature[
                [names[0], names[1], names[2], names[3], names[4], names[5], names[6], names[7], names[8], names[9],
                 names[10], names[11]]],
            y_pred=gold_standard[pathology])

        features = explanation_dnn["feature"].unique()
        for i, feat in enumerate(features):
            type_ecg.append(feat)
            feature_list.append(function_name)
            taus = explanation_dnn.query(f'feature == "{feat}"')["tau"].values
            y_dnn = explanation_dnn.query(f'feature == "{feat}"')["influence"].values
            y_gold_standard = explanation_gold_standard.query(f'feature == "{feat}"')["influence"].values
            p_mean_dnn = round(y_dnn[np.where(taus == 0.)][0], 3)
            p_high_dnn = round((y_dnn[np.where(taus == -tau_lim)][0] - p_mean_dnn) / p_mean_dnn, 2)
            p_low_dnn = round((y_dnn[np.where(taus == tau_lim)][0] - p_mean_dnn) / p_mean_dnn, 2)
            p_mean_gs = round(y_gold_standard[np.where(taus == 0.)][0], 3)
            p_high_gs = round((y_gold_standard[np.where(taus == -tau_lim)][0] - p_mean_gs) / p_mean_gs, 2)
            p_low_gs = round((y_gold_standard[np.where(taus == tau_lim)][0] - p_mean_gs) / p_mean_gs, 2)
            proba_high_dnn.append(p_high_dnn)
            proba_low_dnn.append(p_low_dnn)
            proba_mean_dnn.append(p_mean_dnn)
            proba_high_gs.append(p_high_gs)
            proba_low_gs.append(p_low_gs)
            proba_mean_gs.append(p_mean_gs)

    df = pd.DataFrame({"Feature": feature_list,
                       "ECG": type_ecg,
                       "DNN - Avg P": proba_mean_dnn,
                       "DNN - Ecart relatif en -" + str(tau_lim): proba_low_dnn,
                       "DNN - Ecart relatif en +" + str(tau_lim): proba_high_dnn,
                       "GS - Avg P": proba_mean_gs,
                       "GS - Ecart relatif en -" + str(tau_lim): proba_low_gs,
                       "GS - Ecart relatif en +" + str(tau_lim): proba_high_gs,
                       })

    return df.set_index(["ECG", "Feature"])


# save files as xlsx
for disease in ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']:
    print(disease)
    explanation = explain_mono_feature_influence(disease)
    explanation.to_excel("affichage_ecg/mono_feature_explanation_" + disease + "_2.xlsx")


def explain_poly_feature_influence(pathology, mono_features_list=poly_features, tau_lim=0.7):
    feature_list = []
    proba_mean_dnn = []
    proba_low_dnn = []
    proba_high_dnn = []
    proba_mean_gs = []
    proba_low_gs = []
    proba_high_gs = []

    for feat_function in mono_features_list:
        """Computes the feature values matrix"""
        feature_values = np.zeros(N_patients)
        for id in range(N_patients):
            feature_values[id] = feat_function(table_ecg[id])

        """extracts the name of the function"""
        function_name = str(feat_function)[10:]
        i = 0
        while function_name[i] != " ":
            i += 1
        function_name = function_name[:i]

        """Builds the feature dataframe"""
        series = {
            function_name: feature_values
        }
        df_feature = pd.DataFrame(series)

        explanation_dnn = explainer.explain_influence(
            X_test=df_feature[function_name],
            y_pred=dnn_annotations[pathology])
        explanation_gold_standard = explainer.explain_influence(
            X_test=df_feature[function_name],
            y_pred=gold_standard[pathology])

        features = explanation_dnn["feature"].unique()
        for i, feat in enumerate(features):
            feature_list.append(function_name)
            taus = explanation_dnn.query(f'feature == "{feat}"')["tau"].values
            y_dnn = explanation_dnn.query(f'feature == "{feat}"')["influence"].values
            y_gold_standard = explanation_gold_standard.query(f'feature == "{feat}"')["influence"].values
            p_mean_dnn = round(y_dnn[np.where(taus == 0.)][0], 3)
            p_high_dnn = round((y_dnn[np.where(taus == -tau_lim)][0] - p_mean_dnn) / p_mean_dnn, 2)
            p_low_dnn = round((y_dnn[np.where(taus == tau_lim)][0] - p_mean_dnn) / p_mean_dnn, 2)
            p_mean_gs = round(y_gold_standard[np.where(taus == 0.)][0], 3)
            p_high_gs = round((y_gold_standard[np.where(taus == -tau_lim)][0] - p_mean_gs) / p_mean_gs, 2)
            p_low_gs = round((y_gold_standard[np.where(taus == tau_lim)][0] - p_mean_gs) / p_mean_gs, 2)
            proba_high_dnn.append(p_high_dnn)
            proba_low_dnn.append(p_low_dnn)
            proba_mean_dnn.append(p_mean_dnn)
            proba_high_gs.append(p_high_gs)
            proba_low_gs.append(p_low_gs)
            proba_mean_gs.append(p_mean_gs)

    df = pd.DataFrame({"Feature": feature_list,
                       "DNN - Avg P": proba_mean_dnn,
                       "DNN - Ecart relatif en -" + str(tau_lim): proba_low_dnn,
                       "DNN - Ecart relatif en +" + str(tau_lim): proba_high_dnn,
                       "GS - Avg P": proba_mean_gs,
                       "GS - Ecart relatif en -" + str(tau_lim): proba_low_gs,
                       "GS - Ecart relatif en +" + str(tau_lim): proba_high_gs,
                       })

    return df.set_index(["Feature"])

# save files as xlsx
# for disease in ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']:
#    print(disease)
#    explanation = explain_poly_feature_influence(disease)
#    explanation.to_excel("affichage_ecg/poly_feature_explanation_" + disease + ".xlsx")


# print(ft.asynchrony(table_ecg[10, 2:4], plot=True))

# plot_mono_feature_influence_dnn(ft.auto_correlation_function(lag=340),
#                                name_of_function='auto correlation for a single heartbeat at 70 bpm', save=1)
# plot_mono_feature_influence_dnn(ft.auto_correlation_function(lag=200),
#                                name_of_function='auto correlation for a single heartbeat at 120 bpm', save=1)
#
# plot_mono_feature_influence_gold(ft.auto_correlation_function(lag=340),
#                                 name_of_function='auto correlation for a single heartbeat at 70 bpm', save=1)
# plot_mono_feature_influence_gold(ft.auto_correlation_function(lag=200),
#                                 name_of_function='auto correlation for a single heartbeat at 120 bpm', save=1)


# plot_poly_feature_influence_dnn(lambda list_ecg: ft.average_asynchrony(list_ecg, ecg_comparaison=table_ecg[0, 0]),
#                                name_of_function='asynchrony ECG 0 et 1')

# print(ft.average_asynchrony(table_ecg[0], ecg_comparaison=table_ecg[0, 0]))
