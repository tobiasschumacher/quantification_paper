from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn import metrics


# ==============================================================================
# Helper Function to load quantifiers
# ==============================================================================

def load_class(module_name, class_name):
    m = import_module(module_name)
    c = getattr(m, class_name)
    return c


# ==============================================================================
# Set up variables for paths and initialize them
# ==============================================================================

DATA_PATH = "data/"

RESULTS_PATH = "results/"
Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

RAW_RESULT_FILES_PATH = f"{RESULTS_PATH}raw/"
Path(RAW_RESULT_FILES_PATH).mkdir(parents=True, exist_ok=True)

CLASSIFIER_TUNING_RESULTS_PATH = f"{RAW_RESULT_FILES_PATH}clf/"
Path(CLASSIFIER_TUNING_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

SVMPERF_PATH = 'svm_perf/'

QFOREST_PATH = "qforest/"
QFOREST_COLUMN_NAMES = ["QF", "QF-AC"]

PREPROCESSED_RESULTS_PATH = f"{RESULTS_PATH}preprocessed/"
Path(PREPROCESSED_RESULTS_PATH).mkdir(parents=True, exist_ok=True)

RESULT_TABLES_PATH = f"{RESULTS_PATH}tables/"
Path(RESULT_TABLES_PATH).mkdir(parents=True, exist_ok=True)

RESULT_TABLES_TEX_PATH = f"{RESULTS_PATH}tables/tex/"
Path(RESULT_TABLES_TEX_PATH).mkdir(parents=True, exist_ok=True)

RESULT_PLOTS_PATH = f"{RESULTS_PATH}plots/"
Path(RESULT_PLOTS_PATH).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Global Variables for all Experiments
# ==============================================================================

# seeds
GLOBAL_SEEDS = [4711, 1337, 42, 90210, 666, 879, 1812, 4055, 711, 512]

# train/test ratios to test against
TRAIN_TEST_RATIOS = [np.array(d) for d in [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]]

TRAINING_DISTRIBUTIONS = dict({
    2: np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.95, 0.05]]),
    3: np.array([[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]]),
    4: np.array([[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]]),
    5: np.array([[0.05, 0.2, 0.1, 0.2, 0.45], [0.05, 0.1, 0.7, 0.1, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])
})

TEST_DISTRIBUTIONS = dict({
    2: np.array(
        [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1],
         [0.95, 0.05], [0.99, 0.01], [1, 0]]),
    3: np.array([[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0., 0.05, 0.95]]),
    4: np.array([[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0, 0, 0.8],
                 [0.3, 0.25, 0.35, 0.1]]),
    5: np.array([[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2],
                 [0.35, 0.05, 0.05, 0.05, 0.5],
                 [0.05, 0.25, 0.15, 0.15, 0.4]])
})

EXPERIMENT_LIST = ["main", "clf", "lequa_main", "lequa_clf", "lequa_opt"]
MAIN_EXPERIMENT_MODES = ["binary", "multiclass"]
MAIN_EXPERIMENT_MEASURES = ["AE", "NKLD"]

# ----------------------------------------------------------------------------------------------------------------------
# Dateset variables
# ----------------------------------------------------------------------------------------------------------------------

PICKLED_DATA_FILE_NAME = "dta.pkl"
PICKLED_BINNED_DATA_FILE_NAME = "dta_binned.pkl"
DATA_PREP_SCRIPT_NAME = "prep.py"

# global data set index
DATASET_INDEX = pd.read_csv(f"{DATA_PATH}/data_index.csv",
                            sep=";",
                            index_col="dataset")

DATASET_LIST = list(DATASET_INDEX.index)

# global quantifier index
QUANTIFIER_INDEX = pd.read_csv("quantifier_index.csv",
                               sep=";",
                               index_col="algorithm")

QUANTIFIER_LIST = list(QUANTIFIER_INDEX.index)

QUANTIFIER_DICT = dict({qf: load_class(QUANTIFIER_INDEX.loc[qf, "module_name"],
                                       QUANTIFIER_INDEX.loc[qf, "class_name"])
                        for qf in QUANTIFIER_LIST})

# ----------------------------------------------------------------------------------------------------------------------
# Column names for result tables
# ----------------------------------------------------------------------------------------------------------------------

RESULT_FILE_CONFIG_COLNAMES = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
RESULT_FILE_COLNAMES_TRAINING_CLASS_PREFIX = "Training_Class_"
RESULT_FILE_COLNAMES_TEST_CLASS_PREFIX = "Test_Class_"
RESULT_FILE_COLNAMES_ABSOLUTE_PREVALENCE_SUFFIX = "_Absolute"
RESULT_FILE_COLNAMES_RELATIVE_PREVALENCE_SUFFIX = "_Relative"
RESULT_FILE_COLNAMES_CLASS_PREDICTIONS_INFIX = "_Prediction_Class_"

PROCESSED_RESULTS_KEY_COLUMNS = ['Seed', 'TT_split', 'D_train', 'D_test', 'dataset', 'Drift_MAE']

# ==============================================================================
# Variables for Main Experiments
# ==============================================================================

DEFAULT_QUANTIFIER_LIST = QUANTIFIER_LIST[:-5]  # exclude SVMperf and Qforest quantifiers from default

SVMPERF_QUANTIFIER_LIST = ["SVM-K", "SVM-Q", "RBF-K", "RBF-Q"]

# default parameters that are shared over multiple quantifiers
DEFAULT_PARAMS_LOGISTIC_REGRESSOR = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class='auto')
DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION = 10
DEFAULT_PARAMS_THRESHOLD_PRECISION = 3
DEFAULT_PARAMS_ITERATOR_EPSILON = 1e-06
DEFAULT_PARAMS_ITERATOR_MAX_ITERATIONS = 1000

QUANTIFIER_DEFAULT_PARAMETER_DICT = {
    "AC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR)
           },
    "PAC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR)
            },
    "TSX": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "predict_proba": True,
            "precision": DEFAULT_PARAMS_THRESHOLD_PRECISION,
            "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
            },
    "TS50": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
             "predict_proba": True,
             "precision": DEFAULT_PARAMS_THRESHOLD_PRECISION,
             "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
             },
    "TSMax": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
              "predict_proba": True,
              "precision": DEFAULT_PARAMS_THRESHOLD_PRECISION,
              "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
              },
    "MS": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
           "predict_proba": True,
           "precision": DEFAULT_PARAMS_THRESHOLD_PRECISION,
           "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
           },
    "GAC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "distance": "L2",
            "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
            "solve_cvx": True
            },
    "GPAC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
             "distance": "L2",
             "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
             "solve_cvx": True
             },
    "DyS": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "predict_proba": True,
            "n_bins": 10,
            "distance": "TS",
            "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
            "solve_cvx": True
            },
    "FMM": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "predict_proba": True,
            "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION
            },
    "readme": {"n_features": None,
               "dist": "L2",
               "solve_cvx": True,
               "n_subsets": 100
               },
    "HDx": {"solve_cvx": True},
    "HDy": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION
            },
    "FM": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
           "distance": "L2",
           "n_folds": DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION,
           "solve_cvx": True
           },
    "ED": {},
    "EM": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
           "eps": DEFAULT_PARAMS_ITERATOR_EPSILON,
           "max_iter": DEFAULT_PARAMS_ITERATOR_MAX_ITERATIONS
           },
    "CDE": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR),
            "eps": DEFAULT_PARAMS_ITERATOR_EPSILON,
            "max_iter": DEFAULT_PARAMS_ITERATOR_MAX_ITERATIONS
            },
    "CC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR)
           },
    "PCC": {"clf": clone(DEFAULT_PARAMS_LOGISTIC_REGRESSOR)
            },
    "SVM-K": {"svmperf_path": SVMPERF_PATH,
              "C": 1
              },
    "SVM-Q": {"svmperf_path": SVMPERF_PATH,
              "C": 1
              },
    "RBF-K": {"svmperf_path": SVMPERF_PATH,
              "C": 1,
              "gamma": 1
              },
    "RBF-Q": {"svmperf_path": SVMPERF_PATH,
              "C": 1,
              "gamma": 1
              },
    "PWK": {"alpha": 1,
            "n_neighbors": 10,
            "algorithm": "auto",
            "metric": "euclidean",
            "leaf_size": 30,
            "p": 2,
            "metric_params": None,
            "n_jobs": None
            },
    "QF": {"qforest_path": QFOREST_PATH,
           "n_trees": 100
           }
}

# ==============================================================================
# Variables for Experiments with Tuned Base Classifiers
# ==============================================================================

# list of datasets to perform quantification on in clf tuning experiments - should have 10k instances at most to keep
# experiments feasible
CLF_DATASET_LIST = DATASET_INDEX[DATASET_INDEX["size"] <= 10000].index

# index of quantifiers that use a base classifier
CLF_QUANTIFIER_LIST = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX.clf_type.isin(["crisp", "prob", "score"])].index)

# CLF_ERROR_RATE_QUANTIFIERS = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX.clf_type == "crisp"].index)
CLF_DECISION_SCORE_QUANTIFIERS = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX.clf_type == "score"].index)
CLF_PROBABILITY_QUANTIFIERS = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX.clf_type == "prob"].index)

N_FOLDS_GRIDSEARCHCV = 5

CLF_TUNING_EXPERIMENT_MODES = MAIN_EXPERIMENT_MODES + ["ovr"]

# Parameters for classifiers to tune
SVC_PARAMS = {'C': [2.0 ** i for i in np.arange(-5, 17, step=2)],
              'gamma': [2.0 ** i for i in np.arange(-17, 7, step=2)]}

RF_PARAMS = {'max_features': [2 ** i for i in np.arange(1, 11, step=1)],
             'min_samples_leaf': [2 ** i for i in np.arange(0, 7, step=1)]}

AB_PARAMS = {'learning_rate': [2.0 ** i for i in np.arange(-19, 5, step=2)]}

LR_PARAMS = {'C': [2.0 ** i for i in np.arange(-15, 17, step=2)],
             'class_weight': [None, "balanced"]}

# Parameters for classifiers to tune
TUNABLE_CLASSIFIER_DICT_CLF_KEY = 'clf'
TUNABLE_CLASSIFIER_DICT_PARAMS_KEY = 'params'

TUNABLE_CLASSIFIER_DICT = {'LR': {TUNABLE_CLASSIFIER_DICT_CLF_KEY: LogisticRegression(solver="lbfgs", max_iter=1000),
                                  TUNABLE_CLASSIFIER_DICT_PARAMS_KEY: LR_PARAMS
                                  },
                           'RF': {TUNABLE_CLASSIFIER_DICT_CLF_KEY: RandomForestClassifier(n_estimators=1000),
                                  TUNABLE_CLASSIFIER_DICT_PARAMS_KEY: RF_PARAMS
                                  },
                           'SV': {TUNABLE_CLASSIFIER_DICT_CLF_KEY: SVC(cache_size=10000),
                                  TUNABLE_CLASSIFIER_DICT_PARAMS_KEY: SVC_PARAMS
                                  },
                           'AB': {TUNABLE_CLASSIFIER_DICT_CLF_KEY: AdaBoostClassifier(n_estimators=100),
                                  TUNABLE_CLASSIFIER_DICT_PARAMS_KEY: AB_PARAMS
                                  }
                           }

TUNABLE_CLASSIFIER_LIST = ["LR", "RF", "AB", "SV"]

TUNABLE_DECISION_SCORE_CLASSIFIERS = ["LR", "SV"]
TUNABLE_PROBABILITY_CLASSIFIERS = ["LR"]

BASE_CLASSIFIER_DICT = {qf: (TUNABLE_DECISION_SCORE_CLASSIFIERS if qf in CLF_DECISION_SCORE_QUANTIFIERS
                             else (TUNABLE_PROBABILITY_CLASSIFIERS if qf in CLF_PROBABILITY_QUANTIFIERS
                                   else TUNABLE_CLASSIFIER_LIST)) for qf in CLF_QUANTIFIER_LIST
                        }

CLF_TUNING_METRIC_DICT = {'accuracy': metrics.accuracy_score,
                          'neg_log_loss': metrics.log_loss,
                          'balanced_accuracy': metrics.balanced_accuracy_score,
                          'f1': metrics.f1_score
                          }

CLF_TUNING_METRICS = list(CLF_TUNING_METRIC_DICT.keys())
CLF_TUNING_BINARY_METRIC = 'accuracy'
CLF_TUNING_OVR_METRIC = 'balanced_accuracy'
CLF_TUNING_MULTICLASS_METRIC = 'accuracy'

# ==============================================================================
# Variables for LeQua Case Study
# ==============================================================================

LEQUA_DATA_PATH = f"{DATA_PATH}lequa/"
LEQUA_RESULT_PATH = f"{RESULTS_PATH}lequa/"
LEQUA_TUNE_PATH = f"{LEQUA_RESULT_PATH}tune/"
LEQUA_PLOT_PATH = f"{RESULT_PLOTS_PATH}lequa/"

Path(LEQUA_RESULT_PATH).mkdir(parents=True, exist_ok=True)
Path(LEQUA_TUNE_PATH).mkdir(parents=True, exist_ok=True)
Path(LEQUA_PLOT_PATH).mkdir(parents=True, exist_ok=True)

LEQUA_DATA_DICT_TRAIN_DATA_KEY = "X_train"
LEQUA_DATA_DICT_TRAIN_LABELS_KEY = "y_train"
LEQUA_DATA_DICT_VAL_PATH_KEY = "val_path"
LEQUA_DATA_DICT_VAL_PREVALENCES_KEY = "val_prevs"
LEQUA_DATA_DICT_TEST_PATH_KEY = "test_path"
LEQUA_DATA_DICT_TEST_PREVALENCES_KEY = "test_prevs"
LEQUA_DATA_DICT_N_CLASSES_KEY = "n_classes"
LEQUA_DATA_DICT_SAMPLE_SIZE_KEY = "sample_size"

LEQUA_BINARY_TRAIN_DATA = pd.read_csv(f"{LEQUA_DATA_PATH}T1A/train/training_data.txt")
LEQUA_BINARY_TRAIN_DICT = {LEQUA_DATA_DICT_TRAIN_DATA_KEY: LEQUA_BINARY_TRAIN_DATA.iloc[:, 1:].to_numpy(),
                           LEQUA_DATA_DICT_TRAIN_LABELS_KEY: LEQUA_BINARY_TRAIN_DATA.label.to_numpy(),
                           LEQUA_DATA_DICT_VAL_PATH_KEY: f"{LEQUA_DATA_PATH}T1A/train/dev_samples/",
                           LEQUA_DATA_DICT_VAL_PREVALENCES_KEY: pd.read_csv(
                               f"{LEQUA_DATA_PATH}T1A/train/dev_prevalences.txt",
                               index_col=0),
                           LEQUA_DATA_DICT_TEST_PATH_KEY: f"{LEQUA_DATA_PATH}T1A/test/test_samples/",
                           LEQUA_DATA_DICT_TEST_PREVALENCES_KEY: pd.read_csv(
                               f"{LEQUA_DATA_PATH}T1A/test/test_prevalences.txt", index_col=0),
                           LEQUA_DATA_DICT_N_CLASSES_KEY: 2,
                           LEQUA_DATA_DICT_SAMPLE_SIZE_KEY: 250
                           }

LEQUA_MULTICLASS_TRAIN_DATA = pd.read_csv("data/lequa/T1B/train/training_data.txt")
LEQUA_MULTICLASS_TRAIN_DICT = {LEQUA_DATA_DICT_TRAIN_DATA_KEY: LEQUA_MULTICLASS_TRAIN_DATA.iloc[:, 1:].to_numpy(),
                               LEQUA_DATA_DICT_TRAIN_LABELS_KEY: LEQUA_MULTICLASS_TRAIN_DATA.label.to_numpy(),
                               LEQUA_DATA_DICT_VAL_PATH_KEY: f"{LEQUA_DATA_PATH}T1B/train/dev_samples/",
                               LEQUA_DATA_DICT_VAL_PREVALENCES_KEY: pd.read_csv(
                                   f"{LEQUA_DATA_PATH}T1B/train/dev_prevalences.txt", index_col=0),
                               LEQUA_DATA_DICT_TEST_PATH_KEY: f"{LEQUA_DATA_PATH}T1B/test/test_samples/",
                               LEQUA_DATA_DICT_TEST_PREVALENCES_KEY: pd.read_csv(
                                   f"{LEQUA_DATA_PATH}T1B/test/test_prevalences.txt", index_col=0),
                               LEQUA_DATA_DICT_N_CLASSES_KEY: 28,
                               LEQUA_DATA_DICT_SAMPLE_SIZE_KEY: 1000
                               }

LEQUA_DATA_DICT = dict({"binary": LEQUA_BINARY_TRAIN_DICT,
                        "multiclass": LEQUA_MULTICLASS_TRAIN_DICT})

LEQUA_EXPERIMENT_IDS = [1, 11, 2, 21, 22, 3]
LEQUA_MAIN_SEED = GLOBAL_SEEDS[0]

LEQUA_MEASURES = ["AE", "NKLD", "RAE"]

LEQUA_NUMBER_OF_BINS_CHOICES = [2, 3, 4, 5, 6, 7, 8]

# ----------------------------------------------------------------------------------------------------------------------
# Variables for Experiments with Tuned Quantifiers
# ----------------------------------------------------------------------------------------------------------------------

TUNABLE_QUANTIFIER_LIST = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX.tunable == 1].index)

TUNED_QUANTIFIER_COLUMN_SUFFIX = "OPT"

LOGISTIC_REGRESSION_BASE_PARAMETER_DICT = {"solver": "lbfgs",
                                           "max_iter": 1000
                                           }

LR_GRID = [LogisticRegression(**{**LOGISTIC_REGRESSION_BASE_PARAMETER_DICT, **params})
           for params in list(ParameterGrid(LR_PARAMS))]

TUNABLE_QUANTIFIER_PARAMETER_GRID_DICT = {
    "AC": {"clf": LR_GRID
           },
    "PAC": {"clf": LR_GRID
            },
    "TSX": {"clf": LR_GRID,
            "predict_proba": [True],
            "precision": [DEFAULT_PARAMS_THRESHOLD_PRECISION],
            "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION],
            },
    "TS50": {"clf": LR_GRID,
             "predict_proba": [True],
             "precision": [DEFAULT_PARAMS_THRESHOLD_PRECISION],
             "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION],
             },
    "TSMax": {"clf": LR_GRID,
              "predict_proba": [True],
              "precision": [DEFAULT_PARAMS_THRESHOLD_PRECISION],
              "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION],
              },
    "MS": {"clf": LR_GRID,
           "predict_proba": [True],
           "precision": [DEFAULT_PARAMS_THRESHOLD_PRECISION],
           "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION],
           },
    "GAC": {"clf": LR_GRID,
            "distance": ["L2"],
            "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
            },
    "GPAC": {"clf": LR_GRID,
             "distance": ["L2"],
             "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
             },
    "DyS": {"clf": LR_GRID,
            "predict_proba": [True],
            "n_bins": [2, 4, 6, 8, 10, 15, 20],
            "distance": ["TS"],
            "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
            },
    "FMM": {"clf": LR_GRID,
            "predict_proba": [True],
            "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
            },
    "readme": {"n_features": [2, 4, 6, 8, 10, 15, 20],
               "n_subsets": [100]
               },
    "HDy": {"clf": LR_GRID,
            "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
            },
    "FM": {"clf": LR_GRID,
           "distance": ["L2"],
           "n_folds": [DEFAULT_PARAMS_NUMBER_FOLDS_IN_CM_ESTIMATION]
           },
    "EM": {"clf": LR_GRID,
           "eps": [DEFAULT_PARAMS_ITERATOR_EPSILON],
           "max_iter": [DEFAULT_PARAMS_ITERATOR_MAX_ITERATIONS]
           },
    "CDE": {"clf": LR_GRID,
            "eps": [DEFAULT_PARAMS_ITERATOR_EPSILON],
            "max_iter": [DEFAULT_PARAMS_ITERATOR_MAX_ITERATIONS]
            },
    "CC": {"clf": LR_GRID
           },
    "PCC": {"clf": LR_GRID
            },
    "PWK": {"n_neighbors": [1, 3, 5, 7, 11, 15, 25, 35, 45],
            "alpha": [1, 2, 3, 4, 5],
            "algorithm": ["auto"],
            "metric": ["euclidean"],
            "leaf_size": [30],
            "p": [2]
            },
    "RBF-K": {"svmperf_path": [SVMPERF_PATH],
              "C": [1],
              "gamma": [2.0 ** i for i in np.arange(-17, 7, step=2)]
              },
    "RBF-Q": {"svmperf_path": [SVMPERF_PATH],
              "C": [1],
              "gamma": [2.0 ** i for i in np.arange(-17, 7, step=2)]
              }
}


# ======================================================================================================================
# PLOTTING-RELATES VARIABES
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions to build variables
# ----------------------------------------------------------------------------------------------------------------------

def get_figsize(columnwidth, wf=0.5, hf=(5. ** 0.5 - 1.0) / 2.0):
    """ Credit: https://stackoverflow.com/a/31527287
    Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hf  # height in inches
    return fig_width, fig_height


def build_color_map(ac_color, cc_color, dm_color):
    cmap = dict()

    for str_qf in QUANTIFIER_LIST:

        if QUANTIFIER_INDEX.loc[str_qf, "module_name"].endswith("AC"):
            curr_color = ac_color
        elif QUANTIFIER_INDEX.loc[str_qf, "module_name"].endswith("matching"):
            curr_color = dm_color
        else:
            curr_color = cc_color

        cmap[str_qf] = curr_color
        if str_qf in CLF_QUANTIFIER_LIST:
            for str_clf in BASE_CLASSIFIER_DICT[str_qf]:
                cmap[f"{str_qf}-{str_clf}"] = curr_color
        if str_qf in TUNABLE_QUANTIFIER_LIST:
            cmap[f"{str_qf}-{TUNED_QUANTIFIER_COLUMN_SUFFIX}"] = curr_color

        if str_qf == "QF":
            cmap["QF-AC"] = CC_COLOR

    return cmap


# ----------------------------------------------------------------------------------------------------------------------
# Actual plotting variables start here
# ----------------------------------------------------------------------------------------------------------------------


PLOTS_BASE_WIDTH = 433.62  # pt
PLOTS_BASE_HEIGHT = get_figsize(PLOTS_BASE_WIDTH, wf=1)[1]

AC_COLOR = "#407FB7"
CC_COLOR = '#57AB27'
DM_COLOR = '#F6A800'
PLOTS_COLOR_MAP = build_color_map(AC_COLOR, CC_COLOR, DM_COLOR)

PLOTS_LIGHT_GRAY = ".8"
PLOTS_DARK_GRAY = ".15"

PLOTS_SNS_DEFAULT_PARAMS = dict({"context": "notebook",
                                 "style": "whitegrid",
                                 "font_scale": 1,
                                 "rc": {'figure.figsize': get_figsize(PLOTS_BASE_WIDTH, wf=1),
                                        "axes.edgecolor": PLOTS_LIGHT_GRAY,
                                        "xtick.color": PLOTS_DARK_GRAY,
                                        "ytick.color": PLOTS_DARK_GRAY,
                                        "xtick.bottom": True,
                                        "xtick.minor.width": 0.5,
                                        "xtick.major.width": 0.5,
                                        "ytick.minor.width": 0.5,
                                        "ytick.major.width": 0.5,
                                        "lines.linewidth": 0.7,
                                        "xtick.major.size": 3,
                                        "ytick.major.size": 3,
                                        "xtick.minor.size": 2,
                                        "ytick.minor.size": 2,
                                        "grid.linewidth": 0.1
                                        }})

# parameters for bigger grids in clf experiments
PLOTS_SNS_BROAD_PLOT_PARAMS = {"context": "notebook",
                               "style": "whitegrid",
                               "font_scale": 1,
                               "rc": {'figure.figsize': get_figsize(PLOTS_BASE_WIDTH, wf=2, hf=(5. ** 0.5 - 1.0) / 4.0),
                                      "axes.edgecolor": PLOTS_LIGHT_GRAY,
                                      "xtick.color": PLOTS_DARK_GRAY,
                                      "ytick.color": PLOTS_DARK_GRAY,
                                      "xtick.bottom": True,
                                      "xtick.minor.width": 0.5, "xtick.major.width": 0.5,
                                      "ytick.minor.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.7,
                                      "xtick.major.size": 3,
                                      "ytick.major.size": 3,
                                      "xtick.minor.size": 2,
                                      "ytick.minor.size": 2,
                                      "grid.linewidth": 0.1
                                      }}

PLOTS_CD_WIDTH, PLOTS_CD_HEIGHT = get_figsize(PLOTS_BASE_WIDTH, wf=1)
PLOTS_SNS_CD_PLOT_PARAMS = {"context": "notebook",
                            "style": "whitegrid",
                            "font_scale": 1,
                            "rc": {"figure.figsize": (PLOTS_CD_WIDTH, PLOTS_CD_HEIGHT),
                                   "axes.edgecolor": PLOTS_LIGHT_GRAY,
                                   "xtick.color": PLOTS_DARK_GRAY,
                                   "ytick.color": PLOTS_DARK_GRAY,
                                   "xtick.bottom": True,
                                   "xtick.minor.width": 0.5, "xtick.major.width": 0.5,
                                   "ytick.minor.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 0.7,
                                   "xtick.major.size": 3,
                                   "ytick.major.size": 3,
                                   "xtick.minor.size": 2,
                                   "ytick.minor.size": 2,
                                   "grid.linewidth": 0.1
                                   }
                            }
