import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
import get_ecg as ge
import read_ecg as re
import ethik
from ethik.cache_explainer import CacheExplainer
from ethik.utils import set_fig_size, to_pandas

""" Testing Ethik """

# Importing the DNN annotations
dnn_annotations = pd.read_csv("./data/annotations/dnn.csv")


# Importing the raw signals
ecg_tracings = h5py.File("./data/ecg_tracings.hdf5", "r")
table_ecg = np.array(ecg_tracings['tracings'])
list_ecg_0lead = table_ecg[:, :, 0]
list_ecg_1lead = table_ecg[:, :, 1]
list_ecg_2lead = table_ecg[:, :, 2]
list_ecg_3lead = table_ecg[:, :, 3]
list_ecg_4lead = table_ecg[:, :, 4]
list_ecg_5lead = table_ecg[:, :, 5]
list_ecg_6lead = table_ecg[:, :, 6]
list_ecg_7lead = table_ecg[:, :, 7]
list_ecg_8lead = table_ecg[:, :, 8]
list_ecg_9lead = table_ecg[:, :, 9]
list_ecg_10lead = table_ecg[:, :, 10]
list_ecg_11lead = table_ecg[:, :, 11]


# Computing the dataframe of averages
avgs_of_leads_0 = np.array([np.mean(re.delete_zeros(list_ecg_0lead[k])) for k in range(np.shape(list_ecg_0lead)[0])])
avgs_of_leads_1 = np.array([np.mean(re.delete_zeros(list_ecg_1lead[k])) for k in range(np.shape(list_ecg_1lead)[0])])
avgs_of_leads_2 = np.array([np.mean(re.delete_zeros(list_ecg_2lead[k])) for k in range(np.shape(list_ecg_2lead)[0])])
avgs_of_leads_3 = np.array([np.mean(re.delete_zeros(list_ecg_3lead[k])) for k in range(np.shape(list_ecg_3lead)[0])])
avgs_of_leads_4 = np.array([np.mean(re.delete_zeros(list_ecg_4lead[k])) for k in range(np.shape(list_ecg_4lead)[0])])
avgs_of_leads_5 = np.array([np.mean(re.delete_zeros(list_ecg_5lead[k])) for k in range(np.shape(list_ecg_5lead)[0])])


series = {
    "moyenne0": avgs_of_leads_0,
    "moyenne1": avgs_of_leads_1,
    "moyenne2": avgs_of_leads_2,
    "moyenne3": avgs_of_leads_3,
    "moyenne4": avgs_of_leads_4,
    "moyenne5": avgs_of_leads_5
}
df_avgs = pd.DataFrame(series)

explainer = ethik.ClassificationExplainer()
fig = explainer.plot_influence(X_test=df_avgs[["moyenne0","moyenne1","moyenne2","moyenne3","moyenne4","moyenne5"]], y_pred=dnn_annotations['SB'],yrange=[0.01,0.03])
fig.show()
