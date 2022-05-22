import numpy as np
import pandas as pd

import os
import argparse
import helpers
from time import localtime, strftime

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from run import train_test_ratios, train_distributions, test_distributions, global_seeds, \
    algorithm_dict, algorithm_index
from tune_clfs import clf_index
from run import data_set_index

# ==============================================================================
# Global Variables
# ==============================================================================

clf_path = "results/raw/clf/"
res_path = "results/raw/"
alg_index = algorithm_index.loc[algorithm_index.clf == 1]

rate_algs = ["CC", "AC", "GAC", "HDy"]
svc_algs = ["TSX", "TS50", "TSMax", "MS", "FormanMM", "DyS"]
prob_algs = ["PCC", "PAC", "GPAC", "FM", "EM", "CDE"]


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=list(alg_index.index), default=None,
        help="Algorithms used in evaluation."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        default=None,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--mc", type=bool, default=True,
        help="Whether or not to run multiclass experiments"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=global_seeds,
        help="Seeds to be used in experiments. By default, all seeds will be used."
    )
    parser.add_argument(
        "--minsize", type=int, nargs="?", default=None,
        help="Minimum size of datasets to consider."
    )
    parser.add_argument(
        "--maxsize", type=int, nargs="?", default=None,
        help="Maximum size of datasets to consider."
    )
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run."
    )
    return parser.parse_args()


def run_clfs(data_sets=None,
             algs=None,
             dt_index=None,
             b_mc=False,
             seeds=global_seeds,
             minsize=None,
             maxsize=None):
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

    if minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= minsize]

    if maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= maxsize]

    if algs is None:
        alg_ind = alg_index
    else:
        alg_ind = alg_index.loc[algs]

    if dt_index is None:
        dt_ratios = train_test_ratios
    else:
        dt_ratios = [train_test_ratios[i] for i in dt_index]

    if not b_mc:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        data_sets = list(df_ind.index)
    else:
        data_sets = list(df_ind.index)

    for dta_name in data_sets:

        n_classes = df_ind.loc[dta_name, "classes"]

        if n_classes > 2:
            algs_passed = list(alg_ind.loc[alg_ind.multiclass > 0].index)
        else:
            algs_passed = list(alg_ind.index)


        # build training and test class distributions
        train_ds = train_distributions[n_classes]

        test_ds = test_distributions[n_classes]

        for seed in seeds:
            # ---- only unbinned data necessary here -----------------------

            tuned_clf_experiments(dta_name,
                                  algs=algs_passed,
                                  dt_ratios=dt_ratios,
                                  train_ds=train_ds,
                                  test_ds=test_ds,
                                  seed=seed)


def tuned_clf_experiments(
        dta_name,
        algs,
        dt_ratios,
        train_ds,
        test_ds,
        seed=4711):
    if len(algs) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=False)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)
    n_cols = 5 + 4 * n_classes + n_classes*(len(set(algs).intersection(set(prob_algs)))
                                            + 2*len(set(algs).intersection(set(svc_algs)))
                                            + 4*len(set(algs).intersection(set(rate_algs))))

    clf_fprefix = "clfs_" + dta_name + "_seed_" + str(seed)
    clf_file = [f for f in os.listdir(clf_path) if clf_fprefix in f][-1]

    clf_matrix = pd.read_csv(clf_path + clf_file, sep=";")
    #acc_cols = [str_clf + "_Best_Accuracy" for str_clf in clf_index.keys()]

    stats_matrix = np.zeros((n_combs, n_cols))

    i = 0

    for dt_distr in dt_ratios:

        for train_distr in train_ds:

            for test_distr in test_ds:

                train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx, dt_distr,
                                                                            train_distr, test_distr, seed)

                print(dt_distr)
                print(train_distr)
                print(test_distr)
                j = len(stats_vec)
                stats_matrix[i, 0:j] = stats_vec

                clf_vec = clf_matrix.iloc[i]

                for str_alg in algs:
                    print(str_alg)
                    if str_alg in svc_algs:
                        clf_list = ["lr", "svc"]
                    elif str_alg in prob_algs:
                        clf_list = ["lr"]
                    else:
                        clf_list = list(clf_index.keys())

                    for str_clf in clf_list:
                        print(str_clf)
                        clf_params = {par: clf_vec.loc[str_clf + "_Best_Param_" + par] for par in
                                      clf_index[str_clf]["params"].keys()}
                        if str_clf == "rf":
                            clf_params = {k: int(v) for k,v in clf_params.items()}
                        clf = clf_index[str_clf]['clf']
                        clf.set_params(**clf_params)

                        p = run_clf_setup(str_alg, X, y, train_index, test_index, clf)
                        print(p)
                        stats_matrix[i, j:(j + n_classes)] = p

                        j += n_classes

                i += 1

    col_names = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
    col_names += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Training_Class_" + str(l) + "_Relative" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Relative" for l in Y]

    for str_alg in algs:
        if str_alg in svc_algs:
            clf_list = ["lr", "svc"]
            for str_clf in clf_list:
                for li in Y:
                    col_names += [str_alg + "_" + str_clf + "_Prediction_Class_" + str(li)]
        elif str_alg in prob_algs:
            clf_list = ["lr"]
            for str_clf in clf_list:
                for li in Y:
                    col_names += [str_alg + "_" + str_clf + "_Prediction_Class_" + str(li)]
        else:
            clf_list = list(clf_index.keys())
            for str_clf in clf_list:
                for li in Y:
                    col_names += [str_alg + "_" + str_clf + "_Prediction_Class_" + str(li)]

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)

    fname = res_path + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".csv"
    stats_data.to_csv(fname, index=False, sep=';')


def run_clf_setup(str_alg,
                  X,
                  y,
                  train_idx,
                  test_idx,
                  clf):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    qf = algorithm_dict[str_alg](clf=clf)

    qf.fit(X_train, y_train)
    p = qf.predict(X_test)

    return p


if __name__ == "__main__":
    args = parse_args()
    run_clfs(args.datasets, args.algorithms, args.dt, args.mc, args.seeds, args.minsize, args.maxsize)
