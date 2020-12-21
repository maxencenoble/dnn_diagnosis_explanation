# Description of the results

## Asynchrony

- `feature_asynchrony.py` : values of the feature `asynchrony` (l1 and l2) calculated for every patient (saved here since it needs a long time... )

## Ecg_simple_visualization

Features some plots of ECG leads.

## Ethik

Contains all the results of influence of features towards the 6 diseases (one folder per disease). Each folder has the same kinds of files : influence plots of features calculated for both DNN and Gold standard (PNG files). All types of ECG are shown on these plots.

## Explanation

This folder aims at giving a numerical explanation of influence features : it contains tables that are calculated from plots in `ethik/` (xlsx files). These tables are created relatively to one disease and give for each feature:
- average probability of disease according to DNN
- average probability of disease according to Gold Standard
- average probability when tau=+/-0.7 according to DNN
- average probability when tau=+/-0.7 according to Gold Standard

Features are split between :

- `mono_feature` : features calculated on 1 lead (12 values per feature)
- `poly_feature` : feature calculated on 12 leads (1 value per feature)
- `asynchrony` : only the feature `asynchrony` (one particular `poly_feature` type)

## FFT

There are some pictures of fft transformations on ECG as well as two folders:
- `threshold greater/`
- `threshold smaller/`
(`sigma` being a parameter linked to the threshold)

They both contain two pieces of information:
- the loss of coefficients due to threshold that was applied
- PCA pictures (% variance explained + plots PCi/PCj)

## Wavelet
Contains several pieces of wavelet analysis :
- `dnn outputs/` : what are the outputs of the DNN towards rebuilt signals (small and greater threshold)
- `jupyter outputs/`: what is depicted as issues in the jupyter notebook `exploration_wavelet.ipynb`
    - `1_parameter_test/`: comparison between global parameters (family, threshold) towards wavelet decomposition (nb of non zero coefficients, error on rebuilt signal)
    - `2_threshold_coeff_test/`: discussion about a global threshold on every coefficient once global parameters are chosen
- `relevance_pca (db5 hard)/` : pictures of PCA (% variance explained + plots PCi/PCj) when parameters are `family=db5` and `type_threshold=hard`
- `sym8 vs db5 (hard)/` : comparison between family parameter towards wavelet decomposition (nb of non zero coefficients, error on rebuilt signal) -> two values of threshold