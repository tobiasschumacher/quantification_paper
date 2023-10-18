import joblib
from itertools import product
from multiprocessing import Pool
import argparse

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV

from QFY.metrics import calc_eps, AE, RAE, NKLD

from helpers import bin_train_data, bin_test_data
from tune_clfs import run_ovr_setup
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
    return parser.parse_args()


# Helper function used to save results from given quantifier on test data to csv file

def save_dict_to_csv(res_dict, key, mode):
    n_classes = LEQUA_DATA_DICT[mode][LEQUA_DATA_DICT_N_CLASSES_KEY]
    df_res = pd.DataFrame(data=res_dict["predictions"],
                          columns=[f"{key}_Prediction_Class_{i}" for i in range(n_classes)])
    for m in ["AE", "RAE", "NKLD"]:
        df_err = pd.DataFrame(data=res_dict[m], columns=[f"{key}_{m}"])
        df_res = pd.concat([df_res, df_err], axis=1)

    df_res.to_csv(f"{LEQUA_RESULT_PATH}{key}_{mode}.csv")


# Helper function to train and test a given quantifier on LeQua data
def quantify_lequa(qf, mode, bin_params, seed, b_forest=False):
    np.random.seed(seed)

    data_dict = LEQUA_DATA_DICT[mode]

    X_train = data_dict[LEQUA_DATA_DICT_TRAIN_DATA_KEY]
    y_train = data_dict[LEQUA_DATA_DICT_TRAIN_LABELS_KEY]

    Y_true = data_dict[LEQUA_DATA_DICT_TEST_PREVALENCES_KEY]
    test_path = data_dict[LEQUA_DATA_DICT_TEST_PATH_KEY]

    sample_size = data_dict[LEQUA_DATA_DICT_SAMPLE_SIZE_KEY]

    res_dict = dict({"qf": qf})
    res_dict["predictions"] = []
    res_dict["AE"] = []
    res_dict["RAE"] = []
    res_dict["NKLD"] = []

    if bin_params is None:

        qf.fit(X_train, y_train)

        # apply quantifiers on test data
        for i in tqdm(range(len(Y_true))):
            fname = f"{test_path}/{i}.txt"

            X_test = pd.read_csv(fname).to_numpy()

            y_true = Y_true.iloc[i].to_numpy()
            y_pred = qf.predict(X_test)

            res_dict["predictions"].append(y_pred)
            res_dict["AE"].append(AE(y_true, y_pred))
            res_dict["RAE"].append(RAE(y_true, y_pred, eps=calc_eps(sample_size)))
            res_dict["NKLD"].append(NKLD(y_true, y_pred, eps=calc_eps(sample_size)))

    else:

        # bin training data
        X_binned, bin_list = bin_train_data(X_train, n_bins=bin_params[0], qcut=bin_params[1])
        qf.fit(X_binned, y_train)

        if not b_forest:

            # apply quantifiers on test data
            for i in tqdm(range(len(Y_true))):
                fname = f"{test_path}{i}.txt"

                X_test = bin_test_data(pd.read_csv(fname).to_numpy(), bin_list)

                y_true = Y_true.iloc[i].to_numpy()
                y_pred = qf.predict(X_test)

                res_dict["predictions"].append(y_pred)
                res_dict["AE"].append(AE(y_true, y_pred))
                res_dict["RAE"].append(RAE(y_true, y_pred, eps=calc_eps(sample_size)))
                res_dict["NKLD"].append(NKLD(y_true, y_pred, eps=calc_eps(sample_size)))

        else:

            ac_dict = dict({"qf": qf})
            ac_dict["predictions"] = []
            ac_dict["AE"] = []
            ac_dict["RAE"] = []
            ac_dict["NKLD"] = []

            # apply quantifiers on test data
            for i in range(len(Y_true)):
                fname = f"{test_path}{i}.txt"

                X_test = bin_test_data(pd.read_csv(fname).to_numpy(), bin_list)

                y_true = Y_true.iloc[i].to_numpy()
                y_pred = qf.predict(X_test)

                cc_pred, ac_pred = y_pred[:len(y_true)], y_pred[len(y_true):]

                res_dict["predictions"].append(cc_pred)
                res_dict["AE"].append(AE(y_true, cc_pred))
                res_dict["RAE"].append(RAE(y_true, cc_pred, eps=calc_eps(sample_size)))
                res_dict["NKLD"].append(NKLD(y_true, cc_pred, eps=calc_eps(sample_size)))

                ac_dict["predictions"].append(ac_pred)
                ac_dict["AE"].append(AE(y_true, ac_pred))
                ac_dict["RAE"].append(RAE(y_true, ac_pred, eps=calc_eps(sample_size)))
                ac_dict["NKLD"].append(NKLD(y_true, ac_pred, eps=calc_eps(sample_size)))

            return res_dict, ac_dict

    return res_dict


def validate_lequa(lambda_qf, params, mode, bin_params, seed):
    np.random.seed(seed)

    data_dict = LEQUA_DATA_DICT[mode]

    X_train = data_dict[LEQUA_DATA_DICT_TRAIN_DATA_KEY]
    y_train = data_dict[LEQUA_DATA_DICT_TRAIN_LABELS_KEY]

    Y_true = data_dict[LEQUA_DATA_DICT_VAL_PREVALENCES_KEY]
    val_path = data_dict[LEQUA_DATA_DICT_VAL_PATH_KEY]

    qf = lambda_qf(**params)

    errors = []

    if bin_params is None:

        qf.fit(X_train, y_train)

        # apply quantifiers on test data
        for i in range(len(Y_true)):
            fname = f"{val_path}{i}.txt"

            X_val = pd.read_csv(fname).to_numpy()

            y_true = Y_true.iloc[i].to_numpy()
            y_pred = qf.predict(X_val)

            errors.append(AE(y_true, y_pred))

    else:

        X_binned, bin_list = bin_train_data(X_train, n_bins=bin_params[0], qcut=bin_params[1])
        qf.fit(X_binned, y_train)

        # apply quantifiers on test data
        for i in range(len(Y_true)):
            fname = f"{val_path}{i}.txt"

            X_val = bin_test_data(X_test=pd.read_csv(fname).to_numpy(), bin_list=bin_list)

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

        rows.append(pd.DataFrame([res_vec], columns=colnames, index=[str_qf]))

    df_res = pd.concat(rows)
    df_res.to_csv(f"{LEQUA_TUNE_PATH}binning_stats_{mode}.csv")


# ----------------------------------------------------------------------------------------------------------------------
# Part b): Actually Apply Quantifiers, Using Best Binning From previous experiments for HDx and readme
# ----------------------------------------------------------------------------------------------------------------------

def parse_optimal_binning(str_qf, mode):
    df_res = pd.read_csv(f"{LEQUA_TUNE_PATH}binning_stats_{mode}.csv", index_col=0)

    min_col = list(df_res)[df_res.loc[str_qf].argmin()]

    n_bins = int(min_col[-1])
    qcut = "q" in min_col

    return n_bins, qcut


def run_main_experiments(quantifiers, mode, seed):
    if quantifiers is None:
        quantifiers = DEFAULT_QUANTIFIER_LIST

    for str_qf in quantifiers:
        qf_params = QUANTIFIER_DEFAULT_PARAMETER_DICT[str_qf]

        print(f"Apply {str_qf} with default parameters on {mode} lequa data")

        if str_qf in list(QUANTIFIER_INDEX.loc[QUANTIFIER_INDEX["continuous"] == 0].index):

            n_bins, qcut = parse_optimal_binning(str_qf, mode)

            if str_qf == "QF" and mode == "binary":
                curr_dict, qf_ac_dict = quantify_lequa(QUANTIFIER_DICT[str_qf](**qf_params),
                                                       mode=mode,
                                                       bin_params=(n_bins, qcut),
                                                       seed=seed,
                                                       b_forest=True)
                save_dict_to_csv(qf_ac_dict, "QF-AC", mode)
            else:
                curr_dict = quantify_lequa(QUANTIFIER_DICT[str_qf](**qf_params),
                                           mode=mode,
                                           bin_params=(n_bins, qcut),
                                           seed=seed)
        else:
            curr_dict = quantify_lequa(QUANTIFIER_DICT[str_qf](**qf_params), mode=mode, bin_params=None, seed=seed)

        save_dict_to_csv(curr_dict, str_qf, mode)


########################################################################################################################
# Experiment 2: Apply Quantifiers with Tuned Base Classifiers
########################################################################################################################

def tune_classifiers(classifiers, mode, cv_loss, n_jobs, seed, save_clf=True):
    data_dict = LEQUA_DATA_DICT[mode]

    X = data_dict[LEQUA_DATA_DICT_TRAIN_DATA_KEY]
    y = data_dict[LEQUA_DATA_DICT_TRAIN_LABELS_KEY]

    if classifiers is None:
        classifiers = TUNABLE_CLASSIFIER_LIST

    if n_jobs is None:
        n_jobs = -1

    np.random.seed(seed)
    for str_clf in classifiers:

        print(str_clf)

        params = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_PARAMS_KEY]

        if str_clf == "RF":
            params["max_features"] = [n for n in params["max_features"] if n < 300]

        if cv_loss is None:
            if mode == "binary":  # and str_clf in TUNABLE_LOG_LOSS_CLASSIFIERS:
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
        df_res.to_csv(f"{LEQUA_TUNE_PATH}{str_clf}_{mode}.csv")

        if save_clf:
            joblib.dump(cv_clf.best_estimator_, f"{LEQUA_TUNE_PATH}{str_clf}_{mode}.joblib")


def tune_ovr_classifiers(classifiers, cv_loss, n_jobs, seed=4711):
    data_dict = LEQUA_DATA_DICT["multiclass"]

    X = data_dict[LEQUA_DATA_DICT_TRAIN_DATA_KEY]
    y = data_dict[LEQUA_DATA_DICT_TRAIN_LABELS_KEY]

    for str_clf in classifiers:
        print(str_clf)
        params = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_PARAMS_KEY]
        scoring = CLF_TUNING_OVR_METRIC if cv_loss is None else cv_loss

        params_list = list(ParameterGrid(params))
        n_configs = len(params_list)

        with Pool(processes=n_jobs) as pool:
            cv_scores = pool.starmap(run_ovr_setup,
                                     [(str_clf,
                                       params,
                                       X,
                                       y,
                                       scoring,
                                       seed) for params in params_list])

        print(f"best {scoring} for {str_clf} is {np.max(cv_scores)}")

        df_results = pd.Series(data=params_list,
                               index=range(n_configs),
                               name="params"
                               ).to_frame()

        df_results["mean_test_score"] = cv_scores
        df_results = df_results.sort_values("mean_test_score", ascending=False)
        df_results["rank_test_score"] = range(1, n_configs + 1)
        df_results = df_results.sort_index()
        df_results.to_csv(f"{LEQUA_TUNE_PATH}{str_clf}_ovr.csv")


# read from cv result csv
def collect_best_classifiers_from_csv(classifiers=TUNABLE_CLASSIFIER_LIST, mode="binary"):
    clf_dict = dict()

    for str_clf in classifiers:
        curr_df = pd.read_csv(f"{LEQUA_TUNE_PATH}{str_clf}_{mode}.csv", index_col=0)
        best_config = curr_df.loc[curr_df.rank_test_score == 1].iloc[0]

        params = dict(eval(best_config.params))

        clf_dict[str_clf] = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
        clf_dict[str_clf].set_params(**params)

    return clf_dict


# read from joblib binaries
def collect_best_classifiers_from_binary(mode):
    clf_dict = dict()
    for str_clf in TUNABLE_CLASSIFIER_LIST:
        clf_dict[str_clf] = joblib.load(f"{LEQUA_TUNE_PATH}{str_clf}_{mode}.joblib")

    return clf_dict


# helper function to run all (or a given list of) quantifiers that require base classifiers
def run_tuned_clf_quantifiers(quantifiers, classifiers, mode, seed):
    tuned_classifier_dict = collect_best_classifiers_from_csv(classifiers=classifiers, mode=mode)
    ovr_classifier_dict = collect_best_classifiers_from_csv(classifiers=classifiers, mode="ovr")

    if quantifiers is None:
        quantifiers = CLF_QUANTIFIER_LIST

    for str_qf in quantifiers:

        classifier_choices = BASE_CLASSIFIER_DICT[str_qf]

        if classifiers is not None:
            clf_list = [clf for clf in classifiers if clf in classifier_choices]
        else:
            clf_list = classifier_choices

        for str_clf in clf_list:

            if mode == "multiclass" and QUANTIFIER_INDEX.loc[str_qf, "multiclass"] == "OVR":
                clf = ovr_classifier_dict[str_clf]
            else:
                clf = tuned_classifier_dict[str_clf]

            if str_clf == "RF":
                clf.set_params(**{"n_jobs": -1})

            qf_params = get_qf_params(str_qf, str_clf)
            qf_params["clf"] = clf

            print(f"Apply {str_qf} with tuned {str_clf} base classifier on {mode} LeQua test data")
            curr_key = f"{str_qf}-{str_clf}"
            curr_dict = quantify_lequa(QUANTIFIER_DICT[str_qf](**qf_params),
                                       mode=mode,
                                       bin_params=None,
                                       seed=seed)

            save_dict_to_csv(curr_dict, curr_key, mode)


########################################################################################################################
# Experiment 3: Apply Quantifiers That Were Tuned on Validation Data
########################################################################################################################

def grid_search_lequa(lambda_qf, param_grid, mode, bin_params, seed, n_jobs=None, return_best_params=False):
    params_list = list(ParameterGrid(param_grid))
    n_configs = len(params_list)

    with Pool(processes=n_jobs) as pool:
        val_scores = pool.starmap(validate_lequa,
                                  [(lambda_qf, params, mode, bin_params, seed) for params in params_list])

    df_results = pd.DataFrame(columns=[f"param_{key}" for key in params_list[0].keys()], index=range(n_configs),
                              data=[[str(val) for val in params.values()] for params in params_list])

    df_results["mean_test_score"] = val_scores
    df_results = df_results.sort_values("mean_test_score")
    df_results["rank_test_score"] = range(1, n_configs + 1)
    df_results = df_results.sort_index()

    if return_best_params:
        return df_results, params_list[np.argmin(val_scores)]

    return df_results


def run_tuned_quantifiers(quantifiers=None, mode="binary", seed=4711):
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

        val_results, best_params = grid_search_lequa(lambda_qf, param_grid, mode, bin_params, seed=seed,
                                                     return_best_params=True)

        val_results.to_csv(f"{LEQUA_TUNE_PATH}{str_qf}_stats.csv")

        print(f"Apply tuned {str_qf} quantifier on {mode} test data.")
        curr_dict = quantify_lequa(QUANTIFIER_DICT[str_qf](**best_params),
                                   mode=mode,
                                   bin_params=bin_params,
                                   seed=seed)

        curr_key = f"{str_qf}-{TUNED_QUANTIFIER_COLUMN_SUFFIX}"
        save_dict_to_csv(curr_dict, curr_key, mode)


if __name__ == "__main__":
    args = parse_args()

    experiments = args.experiments

    for exp_mode in args.modes:

        if 11 in experiments:
            optimize_bins(args.algorithms, mode=exp_mode, seed=args.seed)
        if 1 in experiments:
            run_main_experiments(args.algorithms, exp_mode, args.seed)
        if 21 in experiments:
            tune_classifiers(args.classifiers, exp_mode, args.cv_loss, args.n_jobs, args.seed)
        if 22 in experiments:
            tune_ovr_classifiers(args.classifiers, args.cv_loss, args.n_jobs, args.seed)
        if 2 in experiments:
            run_tuned_clf_quantifiers(args.algorithms, args.classifiers, exp_mode, args.seed)
        if 3 in experiments:
            run_tuned_quantifiers(args.algorithms, exp_mode, args.seed)
