# A Comparison of Quantification Methods

This repository provides the code used in the experiments of our paper

*A Comparison of Quantification Methods*  
*Tobias Schumacher,  Markus Strohmaier, and Florian Lemmerich*



Below we provide a description on how to reproduce our experiments.

## Experiments

All experiments have been run with Python 3.7. An environment including all packages required to run our code is given by the ```environment.yml``` file. Along with standard  packages such as ```numpy```, ```pandas``` or ```scikit-learn```, the ```cvxpy``` package [(link)](https://www.cvxpy.org/) is of particular importance for many algorithms in our experiments.

To reproduce all our experiments, except those running quantification forests and SVMs for quantification , you can simply run our main script via 
```bash
    python3 -m run.py -a {algorithms} -d {datasets} --seeds {seeds} --mc 1
```

where algorithms and datasets to run on can be specified by their respective names as listed in ```alg_index.csv``` and ```data/data_index.csv```. The ```--mc``` argument specifies whether multiclass experiments should be conducted as well. When none of the arguments are specified, all experiments will be executed. The 10 default seeds that we used are specified in ```run.py```

To reproduce the experiments on SVM-based quantification, one first needs to install the [SVMperf](https://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) software by Joachims, and then apply a patch on this code to add quantification-oriented loss functions into the framework, as has been done by Esuli, Moreo and Sebastiani in their [QuaNet](https://github.com/HLT-ISTI/QuaNet) project. 

The path to the compiled SVMperf code must then be provided to the ```svm_path``` variable in the ```run_svm.py``` script, and the experiments can then be run via the command

```bash
    python3 -m run_svm.py -d {datasets} --seeds {seeds}
```



Finally, the experiments on quantification forests can be run via the ```run_forest.py``` script, which however requires the jar-files of the [WEKA](https://www.cs.waikato.ac.nz/ml/weka/)-based implementation of [quantification forests](https://ieeexplore.ieee.org/document/6729537) by Milli et al.



### Algorithms

The implementations of all quantification algorithms, except for quantification forests and SVMperf-based quantification, can be found in the ```QFY```subfolder. This package has been split into three submodules ```adjusted_count```,  ```distribution_matching```, and ```classification_models```, corresponding to the three main categories of algorithms discussed in our paper. An overview on the algorithms, and the modules they are found in, is also provided in ```alg_index.csv```.

### Processing the Results

By default, the main scripts ```run.py```, ```run_forest.py```, and ```runSVM.py``` will store the results of their experiments as CSV-files in a subfolder ```results/raw/```.  As this will result in multiple csv-Files per dataset, these files need to be joined together. The directory ```buildR``` contains the code for this procedure. For historical reasons, the scripts in this folder were implemented in [R](https://www.r-project.org/). In this directory, the ```main.R``` script joins all results per dataset into a single CSV, and further computes the corresponding performance scores, as in our case, in terms of the *Absolute Error (AE)* and the *Normalized Kullback-Leibler Divergence* (NKLD). Additional measures that could be applied are implemented in ```metrics.R```.

The preprocessed CSVs are saved into the subfolder ```results/preprocessed/```. 

Further, the ```main.R```script produces tables of average performance scores per dataset as depicted in our paper, which will be stored in ```results/tables/```.  These tables are also needed to produce our plots of average rankings and the corresponding critical differences.



### Plotting the Results

All plots in our paper were generated from the ```plot_results.ipynb``` notebook. It requires that the preprocessing steps discussed above  have been conducted beforehand, with the results saved into the corresponding subfolders of the ```results```directory. The resulting plots are stored in ```results/plots/```.




