# Explainability of DNN outputs towards cardiac pathology diagnosis

This project is a research project made by two students from Ecole Polytechnique : Maxence Noble and Gaspard Sagot, under the supervision of Laurent Risser (PhD) from Toulouse Mathematics Institute.
It aims at explaining the behaviour of one Deep Neural Network, whose results on disease diagnosis were published in 2020 and were actually better than diagnosis made by cardiologists (see `dnn_files/` for more information). 

We particularly focused on using Python package `ethik` (see folder `ethik-master/`, which provides a clear sensitivity analysis of the DNN. To do so, we designed multiple kinds of features towards original data to extract as most information as possible while being not exhaustive at the same time.
These features are either temporal, statistic or wavelet-related. We finally succeeded in finding "interpretable" explanations for diagnosis of every disease (see folder `results/`). We also discuss about other ways to reach this goal, maybe more efficient, but also requiring a lot of work.
All steps of our work and our discussion are described in our [dissertation](https://gitlab.com/maxencenoble/ea_recherche/-/blob/master/dissertation%20(french)/dissertation_Noble_Sagot.pdf), which is only available in French.

## Requirements

This code was tested on Python 3 and needs several specific requirements for ML models and their explanations. Check `requirements.txt`.
The DNN is expected to work with a version of Tensorflow that is not above 1.15, but may work with an upgraded version.


## Data

Type of data : 827 samples of 12 ECG leads (sampled at 400 Hz, each featuring 4096 points)

This data is not directly provided in this Git and needs to downloaded (see [here](https://gitlab.com/maxencenoble/ea_recherche/-/tree/master/dnn_files/data)).

## Folders and files

- `dissertation (french)/` features the dissertation of this project (PDF file, ony in French).
- `dnn_files/` features all the information about the DNN that is studied. Be aware that you'll have to download both test data and models (hdf5 files) to get the results.
- `research_sources/` features four files that were used for our work:
    - `High-Dimensional-Deep-Learning-master/` : particularly used for wavelet analysis
    - `Deep-Learning-comparison/` : compares the DNN outputs to other classic methods
    - `ArticleDNN.pdf` : the article from Nature published to described the performance of the DNN
    - `ArticleEntropicVariableBoosting.pdf` : the article published to described the theoretical aspects of `ethik`
- `utils/` : useful for visualisation of analysis
- `utils_ecg/` : useful to extract ECG data

### About the features

- `features.py` contains a lot of functions to extract information from ECG leads

### About Ethik

- `ethik-master` features all the information needed about `ethik` (use cases, explanations...)
- `run_ethik.py` enables to get explanation of feature influence with either tables or curves

### About Wavelet/FFT analysis

File `exploration_wavelet_fourier.py` enables to get precision of transformation from ECG data to Fourier or Wavelet data.

#### Script for fft analysis
```
python exploration_wavelet_fourier.py fft id_ecg sigma
```
with :
- `id_ecg` : int = 0,...,11 (type of ECG)
- `sigma` : float, ideally `1.`

#### Script for wavelet analysis
```
python exploration_wavelet_fourier.py wavelet id_ecg sigma family level type_threshold
```

with :
- `family` : string, ideally `sym7`,`sym8` or `db5`
- `level` : int, ideally `5`
- `type_threshold` : string, ideally `hard`

### About Lime

File `exploration_lime.py` enables to produce HTML files of explanation towards wavelet analysis. It is designed as to save these files in `output_lime/`, specifically for each disease. An example of what it might be like is also depicted at the root of this folder.
It can be used for both checking influence of certain features or to produce an explanation file about all the patient that positive for one disease.

- `i` : int = 1,...,6 (corresponding to 1dAvb, RBBB, LBBB, SB, AF, ST)
- `e` : float, ideally between 0. and 0.3 (global threshold on wavelet coefficients)
- `j` : int = 0,...,826

#### Script for checking

File has to be modified (see "TO MODIFY" box) to specify the features that need to be checked. 

```
python exploration_lime.py --mode check --id_disease i --threshold_coeff e --id_patient j
```

#### Script for explaining

```
python exploration_lime.py --mode explain --id_disease i --threshold_coeff e
```

### About the results

Folder `results/` provide a specific `README`.

### Exploration of methods

Folder `jupyter_notebooks` contains 4 notebooks which provide figures and pictures to better understand what we did.
- `exploration_ethik.ipynb`
- `exploration_features.ipynb`
- `exploration_periode.ipynb`
- `exploration_wavelet.ipynb`

