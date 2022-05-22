import numpy as np
import pandas as pd

import argparse
import helpers
from time import localtime, strftime

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from run import train_test_ratios, train_distributions, test_distributions, global_seeds
from run import data_set_index

# ==============================================================================
# Global Variables
# ==============================================================================

res_path = "results/raw/clf/"

svc_params = {'C': [2.0 ** i for i in np.arange(-5, 17, step=2)],
              'gamma': [2.0 ** i for i in np.arange(-17, 7, step=2)]}

rf_params = {'max_features': [2 ** i for i in np.arange(1, 11, step=1)],
             'min_samples_leaf': [2 ** i for i in np.arange(0, 7, step=1)]}

ada_params = {'learning_rate': [2.0 ** i for i in np.arange(-19, 5, step=2)]}

lr_params = {'C': [2.0 ** i for i in np.arange(-15, 17, step=2)]
             }

clf_index = {'lr': {'clf': LogisticRegression(solver="lbfgs", max_iter=1000),
                    'params': lr_params},
             'rf': {'clf': RandomForestClassifier(n_estimators=1000),
                    'params': rf_params},
             'svc': {'clf': SVC(cache_size=10000),
                     'params': svc_params},
             'ada': {'clf': AdaBoostClassifier(n_estimators=100),
                     'params': ada_params}}


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=list(clf_index), default=list(clf_index),
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
    parser.add_argument(
        "--njobs", type=int, nargs="?", default=-1,
        help="Number of cores to use in parallel processing"
    )
    return parser.parse_args()


def tune_clfs(data_sets=None,
              clfs=None,
              dt_index=None,
              b_mc=False,
              seeds=global_seeds,
              minsize=None,
              maxsize=None,
              n_jobs=-1):
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

    if minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= minsize]

    if maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= maxsize]

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

        # build training and test class distributions
        train_ds = train_distributions[n_classes]

        test_ds = test_distributions[n_classes]

        for seed in seeds:
            # ---- only unbinned data necessary here -----------------------

            synth_clf_experiments(dta_name,
                                  clfs=clfs,
                                  dt_ratios=dt_ratios,
                                  train_ds=train_ds,
                                  test_ds=test_ds,
                                  seed=seed,
                                  n_jobs=n_jobs)


def synth_clf_experiments(
        dta_name,
        clfs,
        dt_ratios,
        train_ds,
        test_ds,
        seed=4711,
        n_jobs=-1):
    if len(clfs) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=False)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)

    n_cols = 5 + 4 * n_classes + sum(len(clf_index[str_clf]['params']) for str_clf in clfs) + len(clfs)

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

                M = X.shape[1]

                for str_clf in clfs:
                    print(str_clf)

                    params = clf_index[str_clf]['params']

                    if str_clf == "rf":
                        params["max_features"] = [n for n in params["max_features"] if n < M]

                    n_params = len(params)

                    p = tune_setup(clf_index[str_clf]['clf'], params, X, y, train_index, n_jobs)

                    print(p[-1])

                    stats_matrix[i, j:(j + n_params + 1)] = p

                    j += n_params + 1

                i += 1

    col_names = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
    col_names += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Training_Class_" + str(l) + "_Relative" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Relative" for l in Y]

    for alg in clfs:
        for par in clf_index[alg]['params']:
            col_names += [alg + "_Best_Param_" + str(par)]

        col_names += [alg + "_Best_Accuracy_"]

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)

    fname = res_path + "clfs_" + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S",
                                                                                   localtime()) + ".csv"
    stats_data.to_csv(fname, index=False, sep=';')


def tune_setup(clf,
               params,
               X,
               y,
               train_idx,
               n_jobs=-1):
    X_train, y_train = X[train_idx], y[train_idx]

    _, counts = np.unique(y_train, return_counts=True)

    cv_clf = GridSearchCV(clf, params, n_jobs=n_jobs)
    cv_clf.fit(X_train, y_train)

    df_res = pd.DataFrame.from_dict(cv_clf.cv_results_)
    best_config = df_res.loc[df_res.rank_test_score == 1].iloc[0]

    cols = ['param_' + p for p in params] + ['mean_test_score']

    return best_config.loc[cols].to_numpy()


if __name__ == "__main__":
    args = parse_args()
    tune_clfs(args.datasets, args.algorithms, args.dt, args.mc, args.seeds, args.minsize, args.maxsize, args.njobs)
