# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 17:41:20 2019

@author: Tobias Schumacher
"""

import numpy as np
import pandas as pd
import subprocess
from time import localtime, strftime
from run import train_test_ratios, train_distributions, test_distributions, global_seeds, res_path
from run import data_set_index
import helpers
import argparse

qforest_path = "qforest/"


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
        "--mc", type=bool, default=True,
        help="Whether or not to run multiclass experiments"
        )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=global_seeds,
        help="Seeds to be used in experiments. By default, all seeds will be used."
        )
    parser.add_argument(
        "--dt", type=int, nargs="+", default=train_test_ratios,
        help="Index for train/test-splits to be run."
    )
    return parser.parse_args()


def run_forest(data_sets=None,
               dt_ratios=None,
               b_mc=False,
               seed=4711):
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

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

        print(dta_name)
        X, y, N, Y, n_classes, y_cts, y_idx = getxy(dta_name, load_from_disk=True)

        n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)

        colnames = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
        colnames += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
        colnames += ["Training_Class_" + str(l) + "_Relative" for l in Y]
        colnames += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
        colnames += ["Test_Class_" + str(l) + "_Relative" for l in Y]
        colnames += ["QForestCC_Prediction_Class_" + str(l) for l in Y]

        if n_classes == 2:
            colnames += ["QForestAC_Prediction_Class_" + str(l) for l in Y]

        n_cols = len(colnames)

        stats_matrix = np.zeros((n_combs, n_cols))

        i = 0

        for dt_distr in dt_ratios:

            for train_distr in train_ds:

                for test_distr in test_ds:

                    train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx,
                                                                                    dt_distr, train_distr,
                                                                                    test_distr, seed)

                    print(dt_distr)
                    print(train_distr)
                    print(test_distr)
                    j = len(stats_vec)
                    stats_matrix[i, 0:j] = stats_vec
                    stats_matrix[i, j:] = run_synth(X, y, train_index, test_index, n_classes, seed, dta_name)
                    i += 1

        stats_data = pd.DataFrame(data=stats_matrix,
                                  columns=colnames)

        fname = res_path + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S",
                                                                                  localtime()) + ".csv"
        stats_data.to_csv(fname, index=False, sep=';')


def run_synth(X, y, train_index, test_index, n_classes, seed, dta_name):

    id_str = dta_name + '_' + str(seed)

    train_cmd = 'java -Xmx1G -cp quantify.jar:weka.jar:. weka.classifiers.trees.RandomForest -I 100 -S ' + str(seed) + \
                ' -t train_' + id_str + '.arff -d model_' + id_str

    test_cmd = 'java  -Xmx1G -cp quantify.jar:weka.jar:. weka.classifiers.trees.RandomForest -l  model_' + \
               id_str + ' -T test_' + id_str + '.arff'

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    train_data = X_train.assign(y=pd.Series(y_train).values)
    train_data["y"] = train_data["y"].astype("category")

    test_data = X_test.assign(y=pd.Series(y_test).values)
    test_data["y"] = test_data["y"].astype("category")

    pandas2arff(train_data, qforest_path + "train_" + id_str  + ".arff")
    pandas2arff(test_data, qforest_path + "test_" + id_str + ".arff")

    log = subprocess.run(train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         cwd=qforest_path).stderr.decode('utf-8')
    if len(log) > 1:
        print(log)
        raise ValueError(log)

    log = subprocess.run(test_cmd, shell=True, stdout=subprocess.PIPE, cwd=qforest_path)

    if n_classes > 2:
        cc_stats = parse_mc(log, n_classes)
        print("CC: " + str(cc_stats))
        return cc_stats
    else:
        cc_stats, ac_stats = parse_bin(log)
        if np.any(np.isnan(ac_stats)):
            ac_stats = cc_stats
        else:
            ac_stats = np.clip(ac_stats, 0, 1)
        print("CC: " + str(cc_stats))
        print("AC: " + str(ac_stats))
        return np.concatenate([cc_stats, ac_stats])


def parse_bin(log):
    stats = log.stdout.decode('utf-8').split('=== Confusion Matrix ===\n\n')[1].split('\n')

    cc_stats = stats[4:6]
    cc_stats = [s.split("%")[0] for s in cc_stats]
    cc_stats = np.array([float(s.split(":")[1]) / 100 for s in cc_stats])

    ac_stats = stats[7:9]
    ac_stats = [s.split("%")[0] for s in ac_stats]
    ac_stats = np.array([float(s.split(":")[1]) / 100 for s in ac_stats])

    return cc_stats, ac_stats


def parse_mc(log, n_classes):
    stats = log.stdout.decode('utf-8').split('=== Confusion Matrix ===\n\n')[1].split('\n')

    cc_idx = 2 + n_classes

    cc_stats = stats[cc_idx:cc_idx + n_classes]
    cc_stats = [s.split("%")[0] for s in cc_stats]
    cc_stats = np.array([float(s.split(":")[1]) / 100 for s in cc_stats])

    return cc_stats


def pandas2arff(df, fname, relname="data"):
    def convert_dtype(i):

        dtype = str(dtypes[i])

        if dtype == "category":
            cats = df.iloc[:, i].cat.categories
            return "{" + ','.join(str(c) for c in cats) + "}"
        else:
            return {"int64": "numeric",
                    "float64": "numeric",
                    }.get(dtype, "string")

    csv_str = df.to_csv(header=False, index=False, line_terminator="\n")

    cols = list(df)
    cols = [attr.replace(" ", "_") for attr in cols]
    cols = [attr.replace("'", "") for attr in cols]
    dtypes = [str(d) for d in df.dtypes]
    attr_list = ["@attribute " + str(cols[i]) + " " + convert_dtype(i) for i in range(len(cols))]
    attr_str = "\n".join(attr_list)

    arff_str = "@relation " + str(relname) + "\n" + attr_str + "\n@data\n" + csv_str

    if fname[-5:] != ".arff":
        fname += ".arff"

    arff_file = open(fname, "w")
    arff_file.write(arff_str)
    arff_file.close()


def getxy(dta_name, load_from_disk=True):
    dta = helpers.get_data(dta_name, load_from_disk=load_from_disk, binned=True)

    target = data_set_index.loc[dta_name, "target"]

    X = dta.loc[:, ~dta.columns.isin([target])].astype("category")

    y = dta[target]
    N = len(y)
    y_cts = np.unique(y.values, return_counts=True)
    Y = y_cts[0]
    n_classes = len(Y)
    y_cts = y_cts[1]

    y_idx = [np.where(y == l)[0] for l in Y]

    return X, y.astype("category"), N, Y, n_classes, y_cts, y_idx


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        run_forest(args.datasets, args.dt, args.mc, seed)
