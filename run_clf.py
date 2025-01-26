import argparse

import helpers
from run import run_setup
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
        choices=CLF_QUANTIFIER_LIST, default=CLF_QUANTIFIER_LIST,
        help="quantifiers to used in experiments."
    )
    parser.add_argument(
        "-clfs", "--classifiers", nargs="+", type=str,
        choices=TUNABLE_CLASSIFIER_LIST, default=TUNABLE_CLASSIFIER_LIST,
        help="base classifiers to use in experiments."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        choices=DATASET_LIST, default=CLF_DATASET_LIST,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", choices=MAIN_EXPERIMENT_MODES, default=MAIN_EXPERIMENT_MODES,
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
    return parser.parse_args()


def run_with_tuned_classifiers(datasets,
                               quantifiers,
                               classifiers,
                               dt_index,
                               modes,
                               seeds,
                               minsize,
                               maxsize):
    if len(datasets) == 1 and datasets[0].isnumeric():
        df_ind = DATASET_INDEX.iloc[int(datasets[0]):]
    else:
        df_ind = DATASET_INDEX.loc[datasets]

    if minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= minsize]

    if maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= maxsize]

    quantifier_index = QUANTIFIER_INDEX.loc[quantifiers]

    if dt_index is None:
        dt_ratios = TRAIN_TEST_RATIOS
    else:
        dt_ratios = [TRAIN_TEST_RATIOS[i] for i in dt_index]

    if MULTICLASS_MODE_KEY not in modes:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        datasets = list(df_ind.index)
    elif BINARY_MODE_KEY not in modes:
        df_ind = df_ind.loc[df_ind["classes"] > 2]
        datasets = list(df_ind.index)
    else:
        datasets = list(df_ind.index)

    print(datasets)

    for dta_name in datasets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = TRAINING_DISTRIBUTIONS[n_classes]

        test_ds = TEST_DISTRIBUTIONS[n_classes]

        if n_classes == 2:
            quantifier_list = list(quantifier_index.index)
            for seed in seeds:
                tuned_clf_experiments(dta_name,
                                      quantifiers=quantifier_list,
                                      classifiers=classifiers,
                                      dt_ratios=dt_ratios,
                                      train_ds=train_ds,
                                      test_ds=test_ds,
                                      seed=seed)

        else:
            # in multiclass setting we need to distinguish between ovr and natural multiclass quantifiers,
            # since the base classifiers have different optimization strategies

            ovr_quantifier_list = list(quantifier_index.loc[quantifier_index.multiclass == "OVR"].index)
            mc_quantifier_list = list(quantifier_index.loc[quantifier_index.multiclass == "Yes"].index)

            for seed in seeds:
                tuned_clf_experiments(dta_name,
                                      quantifiers=mc_quantifier_list,
                                      classifiers=classifiers,
                                      dt_ratios=dt_ratios,
                                      train_ds=train_ds,
                                      test_ds=test_ds,
                                      seed=seed,
                                      mode=MULTICLASS_MODE_KEY)

                tuned_clf_experiments(dta_name,
                                      quantifiers=ovr_quantifier_list,
                                      classifiers=classifiers,
                                      dt_ratios=dt_ratios,
                                      train_ds=train_ds,
                                      test_ds=test_ds,
                                      seed=seed,
                                      mode=OVR_MODE_KEY)


def get_qf_params(str_qf, str_clf):
    params = QUANTIFIER_DEFAULT_PARAMETER_DICT[str_qf]
    if str_clf == "SV" and str_qf in CLF_DECISION_SCORE_QUANTIFIERS:
        params["predict_proba"] = False

    return params


def tuned_clf_experiments(
        dta_name,
        quantifiers,
        classifiers,
        dt_ratios,
        train_ds,
        test_ds,
        seed,
        mode=BINARY_MODE_KEY):
    if len(quantifiers) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=False)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)

    clf_matrix_dict = helpers.get_clf_matrices(mode, dta_name, seed, classifiers)

    n_config_cols, col_names = helpers.build_colnames(quantifiers, classifiers=classifiers, experiment="tuned_clf", Y=Y)
    stats_matrix = np.zeros((n_combs, len(col_names)))

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
                stats_matrix[i, 0:j] = stats_vec

                for str_qf in quantifiers:
                    print(str_qf)

                    clf_list = BASE_CLASSIFIER_DICT[str_qf]

                    clf_list = [clf_str for clf_str in clf_list if clf_str in classifiers]

                    for str_clf in clf_list:
                        print(str_clf)
                        clf_vec = clf_matrix_dict[str_clf].iloc[i]
                        clf = TUNABLE_CLASSIFIER_DICT[str_clf][TUNABLE_CLASSIFIER_DICT_CLF_KEY]
                        clf_param_dict = None
                        if mode == OVR_MODE_KEY:
                            clf_param_dict = dict()
                            for yc in Y:
                                curr_params = {
                                    par: clf_vec.loc[f"Class_{yc}_{str_clf}_Best_Param_{par}"]
                                    for par in TUNABLE_CLASSIFIER_DICT[str_clf]["params"].keys()
                                }
                                for par in curr_params.keys():
                                    if curr_params[par] != curr_params[par]:
                                        curr_params[par] = None

                                if str_clf == "RF":
                                    curr_params = {k: int(v) for k, v in curr_params.items()}
                                clf_param_dict[yc] = curr_params

                            qf_params = get_qf_params(str_qf, str_clf)
                            qf_params["clf"] = clf
                        else:

                            clf_params = {par: clf_vec.loc[f"{str_clf}_Best_Param_{par}"]
                                          for par in TUNABLE_CLASSIFIER_DICT[str_clf]["params"].keys()
                                          }

                            for par in clf_params.keys():
                                if clf_params[par] != clf_params[par]:  # hack to check for NaN in a type agnostic way
                                    clf_params[par] = None

                            if str_clf == "RF":
                                clf_params = {k: int(v) for k, v in clf_params.items()}
                            clf.set_params(**clf_params)

                            qf_params = get_qf_params(str_qf, str_clf)
                            qf_params["clf"] = clf

                        p = run_setup(
                            str_qf=str_qf,
                            X_train=X[train_index],
                            y_train=y[train_index],
                            X_test=X[test_index],
                            n_classes=n_classes,
                            params=qf_params,
                            clf_param_dict=clf_param_dict
                        )
                        print(p)
                        stats_matrix[i, j:(j + len(p))] = p

                        j += len(p)

                i += 1

    stats_data = pd.DataFrame(data=stats_matrix, columns=col_names)

    res_file_path = os.path.join(RAW_RESULT_FILES_PATH, RAW_RESULT_FILE_NAME(dta_name, seed))
    stats_data.to_csv(res_file_path, index=False, sep=';')


if __name__ == "__main__":
    args = parse_args()
    run_with_tuned_classifiers(args.datasets, args.algorithms, args.classifiers, args.dt, args.modes, args.seeds,
                               args.minsize, args.maxsize)
