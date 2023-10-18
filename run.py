import argparse
import helpers
from time import localtime, strftime

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
        choices=QUANTIFIER_LIST, default=DEFAULT_QUANTIFIER_LIST,
        help="Quantification algorithms used in evaluation. If none are given, all algorithms except quantification "
             "forests and SVMperf-based quantifiers are used. Forests and SVMperf are not in default, since these "
             "require additional software. Please consult the readme file of this repository for more details about "
             "this issue."
    )
    parser.add_argument(
        "-d", "--datasets", nargs="*", type=str,
        choices=DATASET_LIST, default=DATASET_LIST,
        help="Datasets used in evaluation."
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", choices=MAIN_EXPERIMENT_MODES, default=MAIN_EXPERIMENT_MODES,
        help="Whether to only run experiments on data with binary target labels."
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
        "--seeds", type=int, nargs="+", default=GLOBAL_SEEDS,
        help="Seeds to be used in experiments. By default, all seeds will be used."
    )
    parser.add_argument(
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run."
    )
    return parser.parse_args()


def run_synth(datasets,
              quantifiers,
              dt_index,
              modes,
              seeds,
              minsize,
              maxsize):

    df_ind = DATASET_INDEX.loc[datasets]

    if dt_index is None:
        dt_ratios = TRAIN_TEST_RATIOS
    else:
        dt_ratios = [TRAIN_TEST_RATIOS[i] for i in dt_index]

    if "multiclass" not in modes:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
    elif "binary" not in modes:
        df_ind = df_ind.loc[df_ind["classes"] > 2]

    if minsize is not None:
        df_ind = df_ind.loc[df_ind["size"] >= minsize]

    if maxsize is not None:
        df_ind = df_ind.loc[df_ind["size"] <= maxsize]

    datasets = list(df_ind.index)

    quantifier_index = QUANTIFIER_INDEX.loc[quantifiers]

    mc_quantifiers = list(quantifier_index["multiclass"] != "No")
    mc_quantifiers = list(quantifier_index.loc[mc_quantifiers].index)

    dc_quantifiers = list(quantifier_index["continuous"] == 0)
    dc_quantifiers = list(quantifier_index.loc[dc_quantifiers].index)

    cont_quantifiers = list(quantifier_index["continuous"] == 1)
    cont_quantifiers = list(quantifier_index.loc[cont_quantifiers].index)

    mcc_quantifiers = list(set(mc_quantifiers).intersection(cont_quantifiers))

    mcd_quantifiers = list(set(mc_quantifiers).intersection(dc_quantifiers))

    for dta_name in datasets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = TRAINING_DISTRIBUTIONS[n_classes]

        test_ds = TEST_DISTRIBUTIONS[n_classes]

        for seed in seeds:

            # ------- run on binned data --------------------------------------------
            is_multiclass = n_classes > 2

            if is_multiclass:
                data_synth_experiments(dta_name, binned=True, quantifiers=mcd_quantifiers, dt_ratios=dt_ratios,
                                       train_ds=train_ds, test_ds=test_ds, seed=seed)
            else:
                data_synth_experiments(dta_name, binned=True, quantifiers=dc_quantifiers, dt_ratios=dt_ratios,
                                       train_ds=train_ds, test_ds=test_ds, seed=seed)

            # ----run on unbinned data -----------------------

            if is_multiclass:
                data_synth_experiments(dta_name, binned=False, quantifiers=mcc_quantifiers, dt_ratios=dt_ratios,
                                       train_ds=train_ds, test_ds=test_ds, seed=seed)
            else:
                data_synth_experiments(dta_name, binned=False, quantifiers=cont_quantifiers, dt_ratios=dt_ratios,
                                       train_ds=train_ds, test_ds=test_ds, seed=seed)


def data_synth_experiments(
        dta_name,
        binned,
        quantifiers,
        dt_ratios,
        train_ds,
        test_ds,
        seed):
    if len(quantifiers) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=binned)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)

    n_config_cols, col_names = helpers.build_colnames(quantifiers, experiment="main", Y=Y)
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

                    init_args = QUANTIFIER_DEFAULT_PARAMETER_DICT[str_qf]

                    if str_qf in SVMPERF_QUANTIFIER_LIST:
                        if len(train_index) > 10000:
                            init_args["C"] = 0.1
                    elif str_qf == "QF":
                        init_args["id_str"] = dta_name + '_' + str(seed)
                        init_args["seed"] = seed
                        
                    p = run_setup(str_qf, X[train_index], y[train_index], X[test_index], init_args)

                    print(p)

                    stats_matrix[i, j:(j + len(p))] = p

                    j += len(p)

                i += 1

    stats_data = pd.DataFrame(data=stats_matrix, columns=col_names)

    fname = f"{RAW_RESULT_FILES_PATH}{dta_name}_seed_{seed}_{strftime('%Y-%m-%d_%H-%M-%S', localtime())}.csv"
    stats_data.to_csv(fname, index=False, sep=';')


def run_setup(str_qf,
              X_train,
              y_train,
              X_test,
              params=None):

    if params is None:
        params = {}

    qf = QUANTIFIER_DICT[str_qf](**params)

    qf.fit(X_train, y_train)
    p = qf.predict(X_test)

    if len(params) > 0 and "svmperf_path" in list(params.keys()):
        qf.cleanup()

    return p


if __name__ == "__main__":
    args = parse_args()
    run_synth(args.datasets, args.algorithms, args.dt, args.modes, args.seeds, args.minsize, args.maxsize)
