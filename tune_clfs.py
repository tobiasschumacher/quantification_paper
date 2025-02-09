import argparse

import helpers

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV

from helpers import build_clf_colnames
from config import *


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=TUNABLE_CLASSIFIER_LIST, default=TUNABLE_CLASSIFIER_LIST,
        help="Classifiers that are to be tuned."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        choices=DATASET_LIST, default=DATASET_LIST,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", choices=CLF_TUNING_EXPERIMENT_MODES, default=CLF_TUNING_EXPERIMENT_MODES,
        help="Whether to only run experiments on data with binary target labels."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=GLOBAL_SEEDS,
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
        "--binary_loss", type=str, nargs="?", default=CLF_TUNING_BINARY_METRIC, choices=CLF_TUNING_METRICS,
        help="Metric to optimize binary classifiers on in grid search."
    )
    parser.add_argument(
        "--ovr_loss", type=str, nargs="?", default=CLF_TUNING_OVR_METRIC, choices=CLF_TUNING_METRICS,
        help="Metric to optimize classifiers on in grid search on multiclass data with one-vs-rest approach."
    )
    parser.add_argument(
        "--multiclass_loss", type=str, nargs="?", default=CLF_TUNING_MULTICLASS_METRIC, choices=CLF_TUNING_METRICS,
        help="Metric to optimize classifiers on in grid search on multiclass data."
    )
    parser.add_argument(
        "--n_jobs", type=int, nargs="?", default=None,
        help="Number of cores to use in parallel processing"
    )
    return parser.parse_args()


def tune_classifiers(dataset,
                     classifiers,
                     dt_index,
                     modes,
                     seeds,
                     minsize,
                     maxsize,
                     binary_loss,
                     ovr_loss,
                     multiclass_loss,
                     n_jobs):
    if dataset is None:
        df_ind = DATASET_INDEX
    else:
        df_ind = DATASET_INDEX.loc[dataset]

    if minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= minsize]

    if maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= maxsize]

    if dt_index is None:
        dt_ratios = TRAIN_TEST_RATIOS
    else:
        dt_ratios = [TRAIN_TEST_RATIOS[i] for i in dt_index]

    if MULTICLASS_MODE_KEY not in modes and OVR_MODE_KEY not in modes:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        datasets = list(df_ind.index)
    elif BINARY_MODE_KEY not in modes:
        df_ind = df_ind.loc[df_ind["classes"] > 2]
        datasets = list(df_ind.index)
    else:
        datasets = list(df_ind.index)

    for dta_name in datasets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = TRAINING_DISTRIBUTIONS[n_classes]

        test_ds = TEST_DISTRIBUTIONS[n_classes]

        for seed in seeds:
            # ---- binned data does not need to be considered, but proper handling of multiclass setting is crucial

            if n_classes == 2:
                synth_clf_experiments(dta_name,
                                      classifiers=classifiers,
                                      dt_ratios=dt_ratios,
                                      train_ds=train_ds,
                                      test_ds=test_ds,
                                      cv_loss=binary_loss,
                                      mode=BINARY_MODE_KEY,
                                      seed=seed,
                                      n_jobs=n_jobs)

            else:

                if MULTICLASS_MODE_KEY in modes:
                    synth_clf_experiments(dta_name,
                                          classifiers=classifiers,
                                          dt_ratios=dt_ratios,
                                          train_ds=train_ds,
                                          test_ds=test_ds,
                                          cv_loss=multiclass_loss,
                                          mode=MULTICLASS_MODE_KEY,
                                          seed=seed,
                                          n_jobs=n_jobs)

                if OVR_MODE_KEY in modes:
                    synth_clf_experiments(dta_name,
                                          classifiers=classifiers,
                                          dt_ratios=dt_ratios,
                                          train_ds=train_ds,
                                          test_ds=test_ds,
                                          cv_loss=ovr_loss,
                                          mode=OVR_MODE_KEY,
                                          seed=seed,
                                          n_jobs=n_jobs)


def synth_clf_experiments(
        dta_name,
        classifiers,
        dt_ratios,
        train_ds,
        test_ds,
        cv_loss,
        mode,
        seed,
        n_jobs):
    if len(classifiers) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=False)

    # TODO: adjust for colnames builder for single clf per class in ovr case
    n_config_cols, col_names = build_clf_colnames(classifiers, Y, mode)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)

    stats_matrix = pd.DataFrame(columns=col_names, index=list(range(n_combs)))

    i = 0

    for dt_distr in dt_ratios:

        for train_distr in train_ds:

            for test_distr in test_ds:

                train_index, test_index, stats_vec = helpers.synthetic_draw(N, n_classes, y_cts, y_idx, dt_distr,
                                                                            train_distr, test_distr, seed)

                print(dt_distr)
                print(train_distr)
                print(test_distr)
                j = n_config_cols

                stats_matrix.iloc[i, 0:j] = stats_vec

                M = X.shape[1]

                for str_clf in classifiers:
                    print(str_clf)

                    params = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_PARAMS_KEY]

                    if str_clf == "RF":
                        params["max_features"] = [n for n in params["max_features"] if n < M]

                    n_params = len(params)

                    if mode == OVR_MODE_KEY:

                        lb = LabelBinarizer()
                        yb = lb.fit_transform(y)

                        for ic in range(len(lb.classes_)):
                            yl = yb[:, ic]
                            p = tune_setup(TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY],
                                           params, X[train_index], yl[train_index], cv_loss, seed, n_jobs)

                            stats_matrix.iloc[i, j:(j + n_params + 1)] = p

                            j += n_params + 1

                    else:
                        p = tune_setup(TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY],
                                       params, X[train_index], y[train_index], cv_loss, seed, n_jobs)

                        print(p[-1])

                        stats_matrix.iloc[i, j:(j + n_params + 1)] = p

                        j += n_params + 1

                i += 1

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)

    res_file_path = os.path.join(CLASSIFIER_TUNING_RESULTS_PATH,
                                 CLASSIFIER_TUNING_RESULTS_FILE_NAME(mode, dta_name, seed))
    stats_data.to_csv(res_file_path, index=False, sep=';')


def grid_search_no_folds(clf, params, cv_loss, X, y):
    scores = []
    parameter_list = list(ParameterGrid(params))
    for param_set in parameter_list:
        clf.set_params(**param_set)
        clf.fit(X, y)

        if cv_loss == "neg_log_loss":
            y_pred = clf.predict_proba(X)
            scores.append(-CLF_TUNING_METRIC_DICT[cv_loss](y, y_pred))
        else:
            y_pred = clf.predict(X)
            scores.append(CLF_TUNING_METRIC_DICT[cv_loss](y, y_pred))

    best_ind = np.argmax(scores)
    best_params = parameter_list[best_ind]

    return pd.Series(list(best_params.values()) + [np.max(scores)],
                     index=['param_' + p for p in params] + ['mean_test_score']).to_numpy()


def tune_setup(clf,
               params,
               X,
               y,
               cv_loss,
               seed,
               n_jobs):
    if n_jobs is None:
        n_jobs = -1

    _, counts = np.unique(y, return_counts=True)

    np.random.seed(seed)
    n_folds = min(min(counts), N_FOLDS_GRIDSEARCHCV)

    if n_folds == 1:
        return grid_search_no_folds(clf, params, cv_loss, X, y)

    cv_clf = GridSearchCV(clf, params, scoring=cv_loss, n_jobs=n_jobs, cv=n_folds)
    cv_clf.fit(X, y)

    df_res = pd.DataFrame.from_dict(cv_clf.cv_results_)
    best_config = df_res.loc[df_res.rank_test_score == 1].iloc[0]

    cols = ['param_' + p for p in params] + ['mean_test_score']

    return best_config.loc[cols].to_numpy()


if __name__ == "__main__":
    args = parse_args()
    tune_classifiers(args.datasets, args.algorithms, args.dt, args.modes, args.seeds, args.minsize, args.maxsize,
                     args.binary_loss, args.ovr_loss, args.multiclass_loss, args.n_jobs)
