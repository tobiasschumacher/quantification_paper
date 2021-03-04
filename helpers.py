# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 01:18:20 2018

@author: tobi_
"""

import sys
import importlib.util
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from random import sample
import math

# global data set index
data_set_index = pd.read_csv("data/data_index.csv",
                             sep=";",
                             index_col="dataset")


def get_data(dta_name, load_from_disk, binned=True):
    path = "data/" + dta_name

    if load_from_disk:
        if binned:
            try:
                return pd.read_pickle(path + "/dta_binned.pkl")
            except FileNotFoundError:
                return pd.read_pickle(path + "/dta.pkl")

        return pd.read_pickle(path + "/dta.pkl")
    else:
        prep_file = path + "/prep.py"
        spec = importlib.util.spec_from_file_location("prep", prep_file)
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)
        if binned:
            return prep.prep_data(binned)
        else:
            return prep.prep_data()


def split_at_feature(dta, feat, val, target, cond="==", check_x=False):
    if cond not in ["==", "<=", "<", ">", ">=", "in", "!="]:
        ValueError("Specified condition is invalid")

    if isinstance(feat, str):
        ccol = dta[feat]
    elif isinstance(feat, int):
        ccol = dta.iloc[:, feat]
    else:
        raise TypeError("Parameter 'feat' needs to be a string or integer.")

    if target is None:
        raise ValueError("Target column has to be specified")

    if isinstance(target, str):
        x = dta.loc[:, ~dta.columns.isin([target])]
        y = dta[target]
        Y = y.unique()
    elif isinstance(target, int):
        x = dta.loc[:, ~dta.columns.isin([dta.columns[target]])]
        y = dta.iloc[:, target]
        Y = y.unique()
    else:
        raise TypeError("Parameter 'target' needs to be a string or integer.")

    if cond == "in":
        if not (isinstance(val, list) or isinstance(val, np.ndarray)):
            raise TypeError("Condition is invalid")

        bmask = ccol.isin(val)

    else:
        bmask = eval("ccol" + cond + "val")

    x_train = x.loc[bmask].values
    x_test = dta.loc[~bmask].values

    if check_x:
        x = x.values
        m = x.shape[1]
        feature_values = [np.unique(x[:, i]) for i in range(m)]

        if not all([all(np.isin(feature_values[i], np.unique(x_train[i]))) for i in range(m)]):
            warnings.warn("Not all bins of all features are represented in training set.")

    y_train = y[bmask].values
    y_test = y[~bmask].values

    if not all(np.isin(Y, np.unique(y_train))):
        raise ValueError("Not all elements of target class are present in training data.")
    if not all(np.isin(Y, np.unique(y_test))):
        warnings.warn("Not all elements of target class are present in test data.")

    return x_train, y_train, x_test, y_test


def split_train_test(dta, train_fraction, target, check_x=False):
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("Training fraction has to be a value between 0 and 1.")

    if isinstance(target, str):
        x = dta.loc[:, ~dta.columns.isin([target])].values
        y = dta[target].values
        Y = np.unique(y)
    elif isinstance(target, int):
        x = dta.loc[:, ~dta.columns.isin([dta.columns[target]])].values
        y = dta.iloc[:, target].values
        Y = np.unique(y)
    else:
        raise TypeError("Parameter 'target' needs to be a string or integer.")

    n = x.shape[0]

    train_size = round(n * train_fraction)
    train_ind = sample(range(n), train_size)
    y_train = y[train_ind]

    if check_x:
        m = x.shape[1]
        feature_values = [np.unique(x[:, i]) for i in range(m)]
        x_train = x[train_ind, :]
        b_X = all([all(np.isin(feature_values[i], np.unique(x_train[i]))) for i in range(m)])

        n_ct = 0

        while (not all(np.isin(Y, np.unique(y_train))) or not b_X) and n_ct < 1000:
            train_ind = sample(range(n), train_size)
            x_train = x[train_ind, :]
            y_train = y[train_ind]
            b_X = all([all(np.isin(feature_values[i], np.unique(x_train[i]))) for i in range(m)])
            n_ct = n_ct + 1

        if n_ct > 999:
            raise ValueError(
                "Data is probably unsplittable with respect to check_x criterion. Consider further binning of X.")
        test_ind = list(set(range(n)) - set(train_ind))

        return x_train, y_train, x[test_ind, :], y[test_ind]

    while not all(np.isin(Y, np.unique(y_train))):
        train_ind = sample(range(n), train_size)
        y_train = y[train_ind]

    test_ind = list(set(range(n)) - set(train_ind))
    return x[train_ind, :], y_train, x[test_ind, :], y[test_ind]


def get_draw_size(y_cts, dt, train_distr, test_distr, C=None):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if C is None:
        C = sum(y_cts)

    constraints = [C] + [y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
                         for i in range(len(y_cts))]

    return np.floor(min(constraints))


def synthetic_draw(n_y, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed=4711):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if len(y_cts) != len(train_distr):
        raise ValueError("Length of training distribution does not match number of classes")

    if len(dt_distr) != 2:
        raise ValueError("Length of train/test-split has to equal 2")

    if not math.isclose(np.sum(dt_distr), 1.0):
        raise ValueError("Elements of train/test-split do not sum to 1")

    if not math.isclose(np.sum(train_distr), 1.0):
        raise ValueError("Elements of train distribution do not sum to 1")

    if not math.isclose(np.sum(test_distr), 1.0):
        raise ValueError("Elements of test distribution do not sum to 1")

    n = get_draw_size(y_cts, dt_distr, train_distr, test_distr, C=n_y)

    train_cts = (n * dt_distr[0] * train_distr).astype(int)
    if min(train_cts) == 0:
        raise ValueError("Under given input distributions a class would miss in training")

    test_cts = (n * dt_distr[1] * test_distr).astype(int)

    # fix seed for reproducibility
    np.random.seed(seed)

    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False) for i in range(n_classes)]
    y_idx = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
    test_list = [np.random.choice(y_idx[i], size=test_cts[i], replace=False) if np.size(y_idx[i]) > 0 else [] for i in
                 range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index = np.concatenate(test_list).astype(int)

    n_train = train_index.shape[0]
    n_test = test_index.shape[0]
    M = n_train + n_test
    r_train = n_train * 1.0 / M
    r_test = n_test * 1.0 / M

    train_ratios = train_cts * 1.0 / n_train
    test_ratios = test_cts * 1.0 / n_test

    stats_vec = np.concatenate(
        [np.array([M, n_train, n_test, r_train, r_test]), train_cts, train_ratios, test_cts, test_ratios])

    return train_index, test_index, stats_vec


def get_xy(dta_name, load_from_disk=True, binned=False):
    dta = get_data(dta_name, load_from_disk=load_from_disk, binned=binned)

    target = data_set_index.loc[dta_name, "target"]

    X = dta.loc[:, ~dta.columns.isin([target])].values
    # scale X wrt to max value so all values are in interval [-1,1] and better convergence behavior is achieved
    if not binned:
        X = MaxAbsScaler().fit_transform(X)

    y = dta[target].values
    N = len(y)
    y_cts = np.unique(y, return_counts=True)
    Y = y_cts[0]
    n_classes = len(Y)
    y_cts = y_cts[1]

    y_idx = [np.where(y == l)[0] for l in Y]

    return X, y, N, Y, n_classes, y_cts, y_idx


def load_class(module_name, class_name):
    m = importlib.import_module(module_name)
    c = getattr(m, class_name)
    return c
