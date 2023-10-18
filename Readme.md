# A Comparitive Evaluation of Quantification Methods

This repository provides the code used in the experiments of our paper

*A Comparitive Evaluation of Quantification Methods*  
*Tobias Schumacher,  Markus Strohmaier, and Florian Lemmerich*

Below we provide a description on how to reproduce our experiments, assuming you have pulled this repository.

In case you use this repository for your research, we would appreciate if you would cite our work:
```
@article{schumacher_quantification_2021,
  title={A comparative evaluation of quantification methods},
  author={Schumacher, Tobias and Strohmaier, Markus and Lemmerich, Florian},
  journal={arXiv preprint arXiv:2103.03223},
  year={2021}
}
```

## Experiments

Regarding the structure of this repository, we would like to note that specifically, the ```quantifier_index.csv``` table and the ```config.py``` script are very central for our experiments. The CSV contains a table of all quantifiers that we use in our experiments, along with their key properties that are releveant to our experiments. The config-script contains all global variables that are relevant to specify our experiments. In our code, all global variables are spelled out in CAPS, whereas all other variable used in our experiments are lowercase.


#### Environments

All experiments have been run with Python 3.7. An environment including all packages required to run (almost) our code is given in the ```quant.yml``` file in the ```envs```folder - only the ```quapy.ipynb``` notebook requires another environment, which can be installed via ```quapy.yml```. Along with standard  packages such as ```numpy```, ```pandas``` or ```scikit-learn```, the ```cvxpy``` package [(link)](https://www.cvxpy.org/) is of particular importance for many algorithms in our experiments.


#### Loading Datasets

By default, our script will assume that all datasets are already prepared and stored on disk. If it is run for the first time, one should set the parameter ```load_from_disk=False``` in line 174 in ```run.py```. For all datasets from Kaggle, it is required that the raw datasets are manually downloaded into their corresponding ``data/`` subfolders.


### Main Experiments

To reproduce our main experiments, except those running quantification forests and SVMs for quantification, you can simply run our main script via 
```bash
    python3 -m run -a {algorithms} -d {datasets} --seeds {seeds}
```

where algorithms and datasets to run on can be specified by their respective names as listed in ```quantifier_index.csv``` and ```data/data_index.csv```. An ```--modes``` argument can be used to specify whether only binary or multiclass experiments should be conducted. When none of the arguments are specified, all experiments will be executed according to the default parameters specified in ```config.py```. These defaults include all datasets and all quantifers, except for the SVMperf-based quantifiers and quantification forests, which require additional code to be run:
1. To reproduce the experiments on SVM-based quantification, one first needs to download the [SVMperf](https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) software by Joachims, and compile it AFTER applying a patch on this code to add quantification-oriented loss functions into the framework, as has been done by Esuli, Moreo and Sebastiani in their [QuaNet](https://github.com/HLT-ISTI/QuaNet) project. The path to the compiled SVMperf code must then be provided to the ```SVMPERF_PATH``` variable in ```config.py```, and then one can safely pass ```SVM-K```, ```SVM-Q```, ```RBF-K```, and ```RBF-Q``` as algorithms to ```run.py```.
2. To run quantification forests, one requires the jar-files of the [WEKA](https://www.cs.waikato.ac.nz/ml/weka/)-based implementation of [quantification forests](https://ieeexplore.ieee.org/document/6729537) by Letizia Milli et al. We were provided with these files by Ms. Milli upon request.


### Experiments with Tuned Base Classifiers

To reproduce our experiments on quantification with tuned base classifiers, one needs to first run the script ```tune_clfs.py```, and afterwards the script ```run_clf.py```. 

The first script applies hyperparameter optimization on all specified datasets, and to reproduce our experiments, one needs to run the command     

```bash
    python3 -m tune_clfs.py -a {classifiers} --seeds {seeds} --maxsize 10000
```

where the latter parameter specifies that we only want to consider datasets with at most 10,000 instances. The results, most importantly including the best hyperparameter configuration for each setting, will be stored in the subfolder```results/raw/clf/```, and are needed in the next step. By running the command

```bash
    python3 -m run_clf.py -a {algorithms} --seeds {seeds} --maxsize 10000
```

all experiments are executed with quantifiers using the optimized base classifiers in each setting.


### Case Study on Datasets from LeQua Challenge

To reproduce these experiments, it is required to load the datasets from tasks 1A and 1B of the [LeQua challenge](https://lequa2022.github.io/), which can be found on [Zenodo](https://zenodo.org/records/6546188). We have created a corresponding subdirector within the ```data``` folder, but any custom location can be set up by editing the corresponding variables in ```config.py```. The script for the the LeQua datasets is divided into three main experiments, namely
1. applying quantifiers using their default parameters,
2. applying quantifiers with tuned base classifiers,
3. applying quantifiers for which the parameters were tuned on the validation data.

For experiment 1, it is required that the best binning strategy for the _HDx, readme_ and _quantifcation forest_ methods is determined (experiment code 11). For experiment 2, the base classifiers need to be tuned first (experiment codes 21 and 22).
To perform the corresponding experiments, one needs to run 

```bash
    python3 -m run_lequa -e {experiment IDs} -a {algorithms} -clfs {classifiers}
```

where specifying no argument will, by default, run all experiments in sequence (but will take a while). 


### Algorithms

The implementations of all quantification algorithms can be found in the ```QFY``` folder. This has been set up as a Python module and can be run independent from our experiments. The main rationale behind the implementation was to mimic the functionality that one may know from ```skcikit-learn```. Thus, every quantifier has a ```fit()``` and ```predict()``` function, and parameters have to be declared upon initialization of a quantifier. Note that the SVMperf-based methods and the quantification forests require additional code to be run, as has been mentioned above.


### Processing the Results

By default, the main scripts ```run.py``` and ```run_clf.py``` will store the results of their experiments as CSV-files in the subfolder ```results/raw/```.  As this will result in multiple csv-Files per dataset, these files need to be joined together. The directory ```process``` contains the code for this procedure. For historical reasons, the scripts in this folder were implemented in [R](https://www.r-project.org/). In this directory, the ```main.R``` script joins all results per dataset into a single CSV, and further computes the corresponding performance scores, as in our case, in terms of the *Absolute Error (AE)* and the *Normalized Kullback-Leibler Divergence* (NKLD). Additional measures that could be applied are implemented in ```metrics.R```. The preprocessed CSVs are saved into the subfolder ```results/preprocessed/```. 

The results from the LeQua case study are stored differently, and do not require additional processing.



### Plotting the Results

All plots in our paper were generated from the ```plot_results.ipynb``` notebook. It requires that the preprocessing steps discussed above have been conducted beforehand, with the results saved into the corresponding subfolders of the ```results```directory. The resulting plots are stored in ```results/plots/```.




