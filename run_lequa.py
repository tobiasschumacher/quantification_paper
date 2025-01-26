import warnings

import joblib
from itertools import product

from joblib import Parallel, delayed
import argparse

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize

from QFY.ovr import OVRQuantifier
from QFY.metrics import calc_eps, AE, RAE, NKLD

from helpers import bin_train_data, bin_test_data
from run_clf import get_qf_params

from config import *


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e", "--experiments", nargs="+", type=int,
        choices=LEQUA_EXPERIMENT_IDS, default=None,
        help="Experiments to conduct. Options are following integers:\n"
             "11: determine best binning for readme, HDx and QF methods in experiment 1.\n"
             "1: apply quantifiers with default parameters.\n"
             "21: tune base classifiers on training data.\n"
             "22: tune base classifiers in ovr fashion on multiclass training data.\n"
             "2: apply quantifiers with tuned base classifiers.\n"
             "3: tune quantifiers on validation data and apply them on test data right after."
    )
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=QUANTIFIER_LIST, default=None,
        help="quantifiers to used in experiments."
    )
    parser.add_argument(
        "-clfs", "--classifiers", nargs="+", type=str,
        choices=TUNABLE_CLASSIFIER_LIST, default=None,
        help="base classifiers to use in experiments."
    )
    parser.add_argument(
        "--cv_loss", type=str, nargs="?", default=None, choices=CLF_TUNING_METRICS,
        help="Metric to optimize classifiers on in grid search."
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", choices=MAIN_EXPERIMENT_MODES, default=MAIN_EXPERIMENT_MODES,
        help="Whether to only run experiments on data with binary target labels."
    )
    parser.add_argument(
        "--seed", type=int, nargs=1, default=LEQUA_MAIN_SEED,
        help="Seed to use in experiments."
    )
    parser.add_argument(
        "--n_jobs", type=int, nargs="?", default=None,
        help="Number of jobs to run in multiprocessing."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether to overwrite existing results."
    )
    return parser.parse_args()


# helper functions to read out train, validation and test data

def get_lequa_train_data(mode):
    train_df = pd.read_csv(LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_TRAIN_DATA_PATH_KEY])
    X_train = train_df.iloc[:, 1:].to_numpy()
    y_train = train_df.label.to_numpy()

    return X_train, y_train


def get_lequa_val_prevalences(mode):
    return pd.read_csv(LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_VAL_PREVALENCES_PATH_KEY], index_col=0)


def get_lequa_val_data_sample(mode, index):
    val_path = LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_VAL_DATA_PATH_KEY]
    return pd.read_csv(os.path.join(val_path, f"{index}.txt")).to_numpy()


def get_lequa_test_prevalences(mode):
    return pd.read_csv(LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_TEST_PREVALENCES_PATH_KEY], index_col=0)


def get_lequa_test_data_sample(mode, index):
    val_path = LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_TEST_DATA_PATH_KEY]
    return pd.read_csv(os.path.join(val_path, f"{index}.txt")).to_numpy()


# Helper functions used to save results from given quantifier on test data to csv file
def get_res_csv_header(key, n_classes):
    header_line = ","
    header_line += ",".join(f"{key}_Prediction_Class_{i}" for i in range(n_classes)) + ","
    header_line += ",".join(f"{key}_{error}" for error in LEQUA_MEASURES) + "\n"
    return header_line


def get_res_csv_line(index, prediction, errors):
    res_line = f"{index},"
    res_line += ",".join([str(p) for p in prediction]) + ","
    res_line += ",".join([str(e) for e in errors]) + "\n"
    return res_line


# Helper function to train and test a given quantifier on LeQua data
def quantify_lequa(qf, mode, bin_params, seed, file_key, b_forest=False, b_overwrite=False):
    np.random.seed(seed)

    n_classes = LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_N_CLASSES_KEY]

    X_train, y_train = get_lequa_train_data(mode)
    Y_true = get_lequa_test_prevalences(mode)

    sample_size = LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_SAMPLE_SIZE_KEY]

    res_file_path = os.path.join(LEQUA_RESULT_PATH, LEQUA_RESULT_FILE_NAME(file_key, mode))

    start_index = 0
    if not b_overwrite and os.path.isfile(res_file_path):
        with open(res_file_path, "rb") as f:
            num_lines = sum(1 for _ in f)
        if num_lines == len(Y_true) + 1:
            warnings.warn(f"Results for {file_key} in {mode} mode already exist and overwrite is false.")
            return
        else:
            start_index = num_lines - 1
    else:
        header_line = get_res_csv_header(file_key, n_classes)
        with open(res_file_path, "w") as res_file:
            res_file.write(header_line)

    if bin_params is None:

        qf.fit(X_train, y_train)

        # apply quantifiers on test data
        for i in tqdm(range(start_index, len(Y_true))):
            X_test = get_lequa_test_data_sample(mode, i)
            y_true = Y_true.iloc[i].to_numpy()
            y_pred = qf.predict(X_test)

            curr_ae = AE(y_true, y_pred)
            curr_rae = RAE(y_true, y_pred, eps=calc_eps(sample_size))
            curr_nkld = NKLD(y_true, y_pred, eps=calc_eps(sample_size))

            with open(res_file_path, "a") as res_file:
                res_file.write(get_res_csv_line(index=i, prediction=y_pred, errors=[curr_ae, curr_rae, curr_nkld]))
    else:

        # bin training data
        X_binned, bin_list = bin_train_data(X_train, n_bins=bin_params[0], qcut=bin_params[1])
        qf.fit(X_binned, y_train)

        if not b_forest:

            # apply quantifiers on test data
            for i in tqdm(range(start_index, len(Y_true))):
                X_test = bin_test_data(get_lequa_test_data_sample(mode, i), bin_list)
                y_true = Y_true.iloc[i].to_numpy()
                y_pred = qf.predict(X_test)

                curr_ae = AE(y_true, y_pred)
                curr_rae = RAE(y_true, y_pred, eps=calc_eps(sample_size))
                curr_nkld = NKLD(y_true, y_pred, eps=calc_eps(sample_size))

                with open(res_file_path, "a") as res_file:
                    res_file.write(get_res_csv_line(index=i, prediction=y_pred, errors=[curr_ae, curr_rae, curr_nkld]))

        else:
            ac_key = "QF-AC"
            ac_file_path = os.path.join(LEQUA_RESULT_PATH, LEQUA_RESULT_FILE_NAME(ac_key, mode))
            if start_index == 0:
                header_line = get_res_csv_header(file_key, n_classes)
                with open(ac_file_path, "a") as res_file:
                    res_file.write(header_line)

            # apply quantifiers on test data
            for i in range(start_index, len(Y_true)):
                X_test = bin_test_data(get_lequa_test_data_sample(mode, i), bin_list)

                y_true = Y_true.iloc[i].to_numpy()
                y_pred = qf.predict(X_test)

                cc_pred, ac_pred = y_pred[:len(y_true)], y_pred[len(y_true):]

                curr_ae = AE(y_true, y_pred)
                curr_rae = RAE(y_true, y_pred, eps=calc_eps(sample_size))
                curr_nkld = NKLD(y_true, y_pred, eps=calc_eps(sample_size))

                res_line = get_res_csv_line(index=i, prediction=cc_pred, errors=[curr_ae, curr_rae, curr_nkld])
                with open(res_file_path, "a") as res_file:
                    res_file.write(res_line)

                curr_ae = AE(y_true, y_pred)
                curr_rae = RAE(y_true, y_pred, eps=calc_eps(sample_size))
                curr_nkld = NKLD(y_true, y_pred, eps=calc_eps(sample_size))

                res_line = get_res_csv_line(index=i, prediction=ac_pred, errors=[curr_ae, curr_rae, curr_nkld])
                with open(ac_file_path, "a") as res_file:
                    res_file.write(res_line)


def validate_lequa(lambda_qf, params, mode, bin_params, seed):
    np.random.seed(seed)

    X_train, y_train = get_lequa_train_data(mode)

    Y_true = get_lequa_val_prevalences(mode)

    qf = lambda_qf(**params)

    errors = []

    if bin_params is None:

        qf.fit(X_train, y_train)

        # apply quantifiers on validation data
        for i in tqdm(range(len(Y_true))):
            X_val = get_lequa_val_data_sample(mode, i)

            y_true = Y_true.iloc[i].to_numpy()
            y_pred = qf.predict(X_val)

            errors.append(AE(y_true, y_pred))

    else:

        X_binned, bin_list = bin_train_data(X_train, n_bins=bin_params[0], qcut=bin_params[1])
        qf.fit(X_binned, y_train)

        # apply quantifiers on validation data
        for i in range(len(Y_true)):
            X_val = bin_test_data(X_test=get_lequa_val_data_sample(mode, i), bin_list=bin_list)

            y_true = Y_true.iloc[i].to_numpy()
            y_pred = qf.predict(X_val)[:len(y_true)]  # restricting to len(y_true) accommodates qforest as well

            errors.append(AE(y_true, y_pred))

    return np.mean(errors)


########################################################################################################################
# Experiment 1: Apply all Quantifiers with Default Parameters on LeQua Data
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Part a): Determine Good Binning for Quantifiers Requiring Binned Data
# ----------------------------------------------------------------------------------------------------------------------
def optimize_bins(quantifiers,
                  mode,
                  bin_numbers=None,
                  qcuts=None,
                  seed=4711):
    if quantifiers is None:
        quantifiers = list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX["continuous"] == 0].index)

    if bin_numbers is None:
        bin_numbers = LEQUA_NUMBER_OF_BINS_CHOICES

    if qcuts is None:
        qcuts = [True, False]

    rows = []

    # train quantifiers
    for str_qf in quantifiers:

        qf_params = QUANTIFIER_DEFAULT_PARAMETER_DICT[str_qf]
        lambda_qf = QUANTIFIER_DICT[str_qf]

        res_vec = np.zeros(len(bin_numbers) * len(qcuts))
        colnames = []
        i = 0

        # this could also be made a starmap
        for n_bins, b_qcut in product(bin_numbers, qcuts):
            print(f"Apply {str_qf} quantifier using {n_bins} bins and qcut={b_qcut}")
            # bin training data
            colnames.append(f"{'q' if b_qcut else ''}{n_bins}")

            res_vec[i] = validate_lequa(lambda_qf, qf_params, mode, bin_params=(n_bins, b_qcut), seed=seed)
            print(res_vec[i])
            i += 1

        rows.append(pd.DataFrame(data=[res_vec], columns=colnames, index=[str_qf]))

    df_res = pd.concat(rows)
    df_res.to_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_BINNING_STATS_FILE_NAME(mode)))


# ----------------------------------------------------------------------------------------------------------------------
# Part b): Actually Apply Quantifiers, Using Best Binning From previous experiments for HDx and readme
# ----------------------------------------------------------------------------------------------------------------------

def parse_optimal_binning(str_qf, mode):
    df_res = pd.read_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_BINNING_STATS_FILE_NAME(mode)), index_col=0)

    min_col = list(df_res)[df_res.loc[str_qf].argmin()]

    n_bins = int(min_col[-1])
    qcut = "q" in min_col

    return n_bins, qcut


def run_main_experiments(quantifiers, mode, seed, b_overwrite):
    if quantifiers is None:
        quantifiers = DEFAULT_QUANTIFIER_LIST

    for str_qf in quantifiers:
        qf_params = QUANTIFIER_DEFAULT_PARAMETER_DICT[str_qf]

        print(f"Apply {str_qf} with default parameters on {mode} lequa data")

        if str_qf in list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX["continuous"] == 0].index):

            n_bins, qcut = parse_optimal_binning(str_qf, mode)

            if str_qf == "QF" and mode == BINARY_MODE_KEY:
                quantify_lequa(QUANTIFIER_DICT[str_qf](**qf_params),
                               mode=mode,
                               bin_params=(n_bins, qcut),
                               seed=seed,
                               file_key=str_qf,
                               b_forest=True,
                               b_overwrite=b_overwrite)
            else:
                quantify_lequa(qf=QUANTIFIER_DICT[str_qf](**qf_params),
                               mode=mode,
                               bin_params=(n_bins, qcut),
                               seed=seed,
                               file_key=str_qf,
                               b_overwrite=b_overwrite)
        else:
            if mode == MULTICLASS_MODE_KEY and QUANTIFIER_INDEX.loc[str_qf, "multiclass"] == "OVR":
                qf = OVRQuantifier(qf=QUANTIFIER_DICT[str_qf](**params))
            else:
                qf = QUANTIFIER_DICT[str_qf](**qf_params)
            quantify_lequa(qf=qf,
                           mode=mode,
                           bin_params=None,
                           seed=seed,
                           file_key=str_qf,
                           b_overwrite=b_overwrite)


########################################################################################################################
# Experiment 2: Apply Quantifiers with Tuned Base Classifiers
########################################################################################################################

def tune_classifiers(classifiers, mode, cv_loss, n_jobs, seed, save_clf=True):
    X, y = get_lequa_train_data(mode)

    if classifiers is None:
        classifiers = TUNABLE_CLASSIFIER_LIST

    np.random.seed(seed)
    for str_clf in classifiers:

        print(str_clf)

        params = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_PARAMS_KEY]

        if str_clf == "RF":
            params["max_features"] = [n for n in params["max_features"] if n < 300]

        if cv_loss is None:
            if mode == BINARY_MODE_KEY:  # and str_clf in TUNABLE_LOG_LOSS_CLASSIFIERS:
                scoring = CLF_TUNING_BINARY_METRIC
            else:
                scoring = CLF_TUNING_MULTICLASS_METRIC
        else:
            scoring = cv_loss

        clf = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
        cv_clf = GridSearchCV(clf, params, verbose=3, n_jobs=n_jobs, scoring=scoring)
        cv_clf.fit(X, y)

        print(f"best {scoring} for {str_clf} is {cv_clf.best_score_}")

        df_res = pd.DataFrame.from_dict(cv_clf.cv_results_)
        df_res.to_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_CLF_TUNE_STATS_FILE_NAME(str_clf, mode)))

        if save_clf:
            joblib.dump(cv_clf.best_estimator_, os.path.join(LEQUA_TUNE_PATH, LEQUA_BEST_CLF_FILE_NAME(str_clf, mode)))


def tune_ovr_classifiers(classifiers, cv_loss, n_jobs, seed):
    X, y = get_lequa_train_data(MULTICLASS_MODE_KEY)
    Y = np.unique(y)
    y = label_binarize(y, classes=Y)
    scoring = CLF_TUNING_OVR_METRIC if cv_loss is None else cv_loss

    np.random.seed(seed)
    for str_clf in classifiers:
        print(str_clf)
        params = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_PARAMS_KEY]

        if str_clf == "RF":
            params["max_features"] = [n for n in params["max_features"] if n < 300]

        for i, yc in enumerate(Y):
            clf = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
            cv_clf = GridSearchCV(clf, params, verbose=3, n_jobs=n_jobs, scoring=scoring)
            cv_clf.fit(X, y[:, i])

            print(f"best {scoring} for {str_clf} when OVR predicting class {yc} is {cv_clf.best_score_}")

            df_res = pd.DataFrame.from_dict(cv_clf.cv_results_)
            df_res.to_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_CLF_TUNE_OVR_CLASS_STATS_FILE_NAME(str_clf, yc)))


# read from cv result csv
def collect_best_classifiers_from_csv(classifiers=TUNABLE_CLASSIFIER_LIST, mode=BINARY_MODE_KEY):
    clf_dict = dict()

    for str_clf in classifiers:
        curr_df = pd.read_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_CLF_TUNE_STATS_FILE_NAME(str_clf, mode)), index_col=0)
        best_config = curr_df.loc[curr_df.rank_test_score == 1].iloc[0]

        params = dict(eval(best_config.params))

        clf_dict[str_clf] = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
        clf_dict[str_clf].set_params(**params)

    return clf_dict


def collect_ovr_param_dict(str_clf):
    X, y = get_lequa_train_data(MULTICLASS_MODE_KEY)
    Y = np.unique(y)

    clf_params_dict = dict()

    for i, yc in enumerate(Y):
        curr_df = pd.read_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_CLF_TUNE_OVR_CLASS_STATS_FILE_NAME(str_clf, yc)),
                              index_col=0)
        best_config = curr_df.loc[curr_df.rank_test_score == 1].iloc[0]

        params = dict(eval(best_config.params))

        clf_params_dict[yc] = params

    return clf_params_dict


# read from joblib binaries
def collect_best_classifiers_from_binary(mode):
    clf_dict = dict()
    for str_clf in TUNABLE_CLASSIFIER_LIST:
        clf_dict[str_clf] = joblib.load(os.path.join(LEQUA_TUNE_PATH, LEQUA_BEST_CLF_FILE_NAME(str_clf, mode)))

    return clf_dict


# function to run all (or a given list of) quantifiers with tuned base classifiers
def run_tuned_clf_quantifiers(quantifiers, classifiers, mode, seed, n_jobs, b_overwrite):
    if classifiers is None:
        classifiers = TUNABLE_CLASSIFIER_LIST

    tuned_classifier_dict = collect_best_classifiers_from_csv(classifiers=classifiers, mode=mode)

    if quantifiers is None:
        quantifiers = CLF_QUANTIFIER_LIST

    for str_qf in quantifiers:

        classifier_choices = BASE_CLASSIFIER_DICT[str_qf]

        if classifiers is not None:
            clf_list = [clf for clf in classifiers if clf in classifier_choices]
        else:
            clf_list = classifier_choices

        for str_clf in clf_list:

            if mode == MULTICLASS_MODE_KEY and QUANTIFIER_INDEX.loc[str_qf, "multiclass"] == "OVR":

                clf = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
                qf_params = get_qf_params(str_qf, str_clf)
                qf_params["clf"] = clf
                clf_params_dict = collect_ovr_param_dict(str_clf)

                # for random forests, it is faster to use parallelization in training/executing the single classifiers,
                # than to parallelize training of class-wise classifiers
                if str_clf == "RF":
                    for yc in clf_params_dict.keys():
                        clf_params_dict[yc]["n_jobs"] = n_jobs
                    qf = OVRQuantifier(qf=QUANTIFIER_DICT[str_qf](**qf_params), clf_param_dict=clf_params_dict)
                else:
                    qf = OVRQuantifier(qf=QUANTIFIER_DICT[str_qf](**qf_params), clf_param_dict=clf_params_dict,
                                       n_jobs=n_jobs)

            else:

                clf = tuned_classifier_dict[str_clf]

                if str_clf == "RF":
                    clf.set_params(**{"n_jobs": -1})

                qf_params = get_qf_params(str_qf, str_clf)
                qf_params["clf"] = clf
                qf = QUANTIFIER_DICT[str_qf](**qf_params)

            print(f"Apply {str_qf} with tuned {str_clf} base classifier on {mode} LeQua test data")
            curr_key = f"{str_qf}-{str_clf}"
            quantify_lequa(qf=qf, mode=mode, bin_params=None, seed=seed, file_key=curr_key, b_overwrite=b_overwrite)


########################################################################################################################
# Experiment 3: Apply Quantifiers That Were Tuned on Validation Data
########################################################################################################################

def grid_search_lequa(lambda_qf, param_grid, mode, bin_params, seed, n_jobs=None, return_best_params=False):
    params_list = list(ParameterGrid(param_grid))
    n_configs = len(params_list)

    if n_jobs is not None:
        val_scores = Parallel(n_jobs=n_jobs)(delayed(validate_lequa)(lambda_qf, params, mode, bin_params, seed)
                                             for params in params_list)
    else:
        val_scores = []
        for params in params_list:
            print(params)
            score = validate_lequa(lambda_qf, params, mode, bin_params, seed)
            val_scores.append(score)

    df_results = pd.DataFrame(columns=[f"param_{key}" for key in params_list[0].keys()], index=range(n_configs),
                              data=[[str(val) for val in params.values()] for params in params_list])

    df_results["mean_test_score"] = val_scores
    df_results = df_results.sort_values("mean_test_score")
    df_results["rank_test_score"] = range(1, n_configs + 1)
    df_results = df_results.sort_index()

    if return_best_params:
        return df_results, params_list[np.argmin(val_scores)]

    return df_results


def grid_search_lequa_ovr(base_qf, param_grid, mode, bin_params, seed, n_jobs=None, return_best_params=False):
    params_list = list(ParameterGrid(param_grid))
    n_configs = len(params_list)

    val_scores = []
    for params in params_list:
        print(params)
        lambda_qf = OVRQuantifier
        qf_params = dict()
        qf_params["qf"] = base_qf(**params)
        qf_params["n_jobs"] = n_jobs
        score = validate_lequa(lambda_qf, qf_params, mode, bin_params, seed)
        val_scores.append(score)

    df_results = pd.DataFrame(columns=[f"param_{key}" for key in params_list[0].keys()], index=range(n_configs),
                              data=[[str(val) for val in params.values()] for params in params_list])

    df_results["mean_test_score"] = val_scores
    df_results = df_results.sort_values("mean_test_score")
    df_results["rank_test_score"] = range(1, n_configs + 1)
    df_results = df_results.sort_index()

    if return_best_params:
        return df_results, params_list[np.argmin(val_scores)]

    return df_results


def run_tuned_quantifiers(quantifiers=None, mode=BINARY_MODE_KEY, n_jobs=None, seed=4711, b_overwrite=False):
    if quantifiers is None:
        quantifiers = TUNABLE_QUANTIFIER_LIST

    for str_qf in quantifiers:

        param_grid = TUNABLE_QUANTIFIER_PARAMETER_GRID_DICT[str_qf]

        print(f"Tune {str_qf} quantifier on {mode} validation data.")
        lambda_qf = QUANTIFIER_DICT[str_qf]

        if str_qf in list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX["continuous"] == 0].index):
            nbins, qcut = parse_optimal_binning(str_qf, mode)
            bin_params = (nbins, qcut)
        else:
            bin_params = None

        if mode == MULTICLASS_MODE_KEY and QUANTIFIER_INDEX.loc[str_qf, "multiclass"] == "OVR":
            val_results, best_params = grid_search_lequa_ovr(
                base_qf=lambda_qf,
                param_grid=param_grid,
                mode=mode,
                bin_params=bin_params,
                n_jobs=n_jobs,
                seed=seed,
                return_best_params=True
            )

        else:
            val_results, best_params = grid_search_lequa(lambda_qf=lambda_qf,
                                                         param_grid=param_grid,
                                                         mode=mode,
                                                         bin_params=bin_params,
                                                         n_jobs=n_jobs,
                                                         seed=seed,
                                                         return_best_params=True)

        val_results.to_csv(os.path.join(LEQUA_TUNE_PATH, LEQUA_VAL_RESULTS_FILE_NAME(str_qf, mode)))

        print(f"Apply tuned {str_qf} quantifier on {mode} test data.")
        curr_key = f"{str_qf}-{TUNED_QUANTIFIER_COLUMN_SUFFIX}"
        quantify_lequa(QUANTIFIER_DICT[str_qf](**best_params),
                       mode=mode,
                       bin_params=bin_params,
                       seed=seed,
                       file_key=curr_key,
                       b_overwrite=b_overwrite)


if __name__ == "__main__":
    args = parse_args()

    experiments = args.experiments

    for exp_mode in args.modes:

        if 11 in experiments:
            optimize_bins(quantifiers=args.algorithms, mode=exp_mode, seed=args.seed)
        if 1 in experiments:
            run_main_experiments(quantifiers=args.algorithms, mode=exp_mode, seed=args.seed, b_overwrite=args.overwrite)
        if 21 in experiments:
            tune_classifiers(classifiers=args.classifiers,
                             mode=exp_mode,
                             cv_loss=args.cv_loss,
                             n_jobs=args.n_jobs,
                             seed=args.seed)
        if 22 in experiments:
            tune_ovr_classifiers(classifiers=args.classifiers, cv_loss=args.cv_loss, n_jobs=args.n_jobs, seed=args.seed)
        if 2 in experiments:
            run_tuned_clf_quantifiers(quantifiers=args.algorithms,
                                      classifiers=args.classifiers,
                                      mode=exp_mode,
                                      seed=args.seed,
                                      n_jobs=args.n_jobs,
                                      b_overwrite=args.overwrite)
        if 3 in experiments:
            run_tuned_quantifiers(quantifiers=args.algorithms,
                                  mode=exp_mode,
                                  n_jobs=args.n_jobs,
                                  seed=args.seed,
                                  b_overwrite=args.overwrite)
