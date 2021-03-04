# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:31:23 2018

@author: tobi_
"""

import pandas as pd
import numpy as np
from QFY.classification_models import SVMPerf
from time import localtime, strftime
import argparse

from run import train_test_ratios, train_distributions, test_distributions, global_seeds, res_path
from run import data_set_index
import helpers

svmperf_path = '../SVMperf/'


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        default=None,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--timeout", type=int,
        help="Timeout in seconds for training individual SVM instance"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=global_seeds,
        help="Seeds to be used in experiments. By default, all seeds will be used."
    )
    return parser.parse_args()


def run_svm(datasets=None,
            timeout=None,
            seed=4711):

    if datasets is None:
        df_ind = data_set_index
    elif len(datasets) == 1 and datasets[0].isnumeric():
        min_ind = int(datasets[0])
        df_ind = data_set_index.iloc[min_ind:]
    else:
        df_ind = data_set_index.loc[datasets]

    df_ind = df_ind.loc[df_ind["classes"] == 2]
    datasets = list(df_ind.index)

    for dta_name in datasets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = train_distributions[n_classes]
        test_ds = test_distributions[n_classes]

        print(dta_name)
        X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True)

        n_combs = len(train_test_ratios) * len(train_ds) * len(test_ds)

        colnames = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
        colnames += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
        colnames += ["Training_Class_" + str(l) + "_Relative" for l in Y]
        colnames += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
        colnames += ["Test_Class_" + str(l) + "_Relative" for l in Y]
        colnames += ["SVM_KLD_Prediction_Class_" + str(l) for l in Y]

        colnames += ["SVM_Q_Prediction_Class_" + str(l) for l in Y]

        n_cols = len(colnames)

        stats_matrix = np.zeros((n_combs, n_cols))

        i = 0

        for dt_distr in train_test_ratios:

            for train_distr in train_ds:

                for test_distr in test_ds:

                    train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx,
                                                                                dt_distr, train_distr,
                                                                                test_distr, seed)

                    X_train, y_train = X[train_index], y[train_index]
                    X_test = X[test_index]

                    print(dt_distr)
                    print(train_distr)
                    print(test_distr)
                    j = len(stats_vec)
                    stats_matrix[i, 0:j] = stats_vec

                    k = 0
                    for loss in ['kld', 'q']:
                        try:
                            stats_matrix[i, (j + k):(j + k + 2)] = train_svm(X_train, y_train, X_test, loss, timeout)
                        except:
                            break
                        k += 2

                    i += 1

        stats_data = pd.DataFrame(data=stats_matrix,
                                  columns=colnames)

        fname = res_path + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S",
                                                                                  localtime()) + ".csv"
        stats_data.to_csv(fname, index=False, sep=';')


def train_svm(X_train,
              y_train,
              X_test,
              loss,
              timeout):
    if len(X_train) > 10000:
        C = 0.1
    else:
        C = 1

    qf = SVMPerf(svmperf_base=svmperf_path, loss=loss, C=C, timeout=timeout)
    qf.fit(X_train, y_train)

    yp = qf.predict(X_test)
    print(yp)
    return yp


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        run_svm(args.datasets, args.timeout, seed)
