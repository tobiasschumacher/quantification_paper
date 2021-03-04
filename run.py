import argparse
import numpy as np
import pandas as pd
import helpers
from time import localtime, strftime
import QFY

# ==============================================================================
# Global Variables
# ==============================================================================

res_path = "results/raw/"

# global data set index
data_set_index = pd.read_csv("data/data_index.csv",
                             sep=";",
                             index_col="dataset")

# global algorithm index
algorithm_index = pd.read_csv("alg_index.csv",
                              sep=";",
                              index_col="algorithm")

algorithm_index = algorithm_index.loc[algorithm_index.export == 1]
algorithms = list(algorithm_index.index)
algorithm_dict = dict({alg: helpers.load_class(algorithm_index.loc[alg, "module_name"],
                                               algorithm_index.loc[alg, "class_name"])
                       for alg in algorithms})

global_seeds = [4711, 1337, 42, 90210, 666, 879, 1812, 4055, 711, 512]

# train/test ratios to test against
train_test_ratios = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]
train_test_ratios = [np.array(d) for d in train_test_ratios]

train_distributions = dict()
train_distributions[2] = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1], [0.95, 0.05]])
train_distributions[3] = np.array([[0.2, 0.5, 0.3], [0.05, 0.8, 0.15], [0.35, 0.3, 0.35]])
train_distributions[4] = np.array([[0.5, 0.3, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
train_distributions[5] = np.array([[0.05, 0.2, 0.1, 0.2, 0.45], [0.05, 0.1, 0.7, 0.1, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])

test_distributions = dict()
test_distributions[2] = np.array(
    [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1],
     [0.95, 0.05], [0.99, 0.01], [1, 0]])
test_distributions[3] = np.array(
    [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0., 0.05, 0.95]])
test_distributions[4] = np.array(
    [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0, 0, 0.8],
     [0.3, 0.25, 0.35, 0.1]])
test_distributions[5] = np.array(
    [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
     [0.05, 0.25, 0.15, 0.15, 0.4]])

mc_data = data_set_index.loc[data_set_index.loc[:, "classes"] > 2].index


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--algorithms", nargs="+", type=str,
        choices=algorithms, default=algorithms,
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
        "--dt", type=int, nargs="+", default=None,
        help="Index for train/test-splits to be run."
    )
    return parser.parse_args()


def run_synth(data_sets=None,
              algs=None,
              dt_index=None,
              b_mc=False,
              seeds=global_seeds):
    if data_sets is None:
        df_ind = data_set_index
    else:
        df_ind = data_set_index.loc[data_sets]

    if dt_index is None:
        dt_ratios = train_test_ratios
    else:
        dt_ratios = [train_test_ratios[i] for i in dt_index]

    if not b_mc:
        df_ind = df_ind.loc[df_ind["classes"] == 2]
        data_sets = list(df_ind.index)
    else:
        data_sets = list(df_ind.index)

    if algs is None:
        alg_ind = algorithm_index
    else:
        alg_ind = algorithm_index.loc[algs]

    mc_algs = list(alg_ind["multiclass"] > 0)
    mc_algs = list(alg_ind.loc[mc_algs].index)

    dc_algs = list(alg_ind["continuous"] == 0)
    dc_algs = list(alg_ind.loc[dc_algs].index)

    cont_algs = list(alg_ind["continuous"] == 1)
    cont_algs = list(alg_ind.loc[cont_algs].index)

    mcc_algs = list(set(mc_algs).intersection(cont_algs))

    mcd_algs = list(set(mc_algs).intersection(dc_algs))

    for dta_name in data_sets:

        n_classes = df_ind.loc[dta_name, "classes"]

        # build training and test class distributions
        train_ds = train_distributions[n_classes]

        test_ds = test_distributions[n_classes]

        for seed in seeds:

            # ------- run on binned data --------------------------------------------
            is_multiclass = n_classes > 2

            if is_multiclass:
                data_synth_experiments(dta_name, binned=True, algs=mcd_algs, dt_ratios=dt_ratios, train_ds=train_ds,
                                       test_ds=test_ds, seed=seed)
            else:
                data_synth_experiments(dta_name, binned=True, algs=dc_algs, dt_ratios=dt_ratios, train_ds=train_ds,
                                       test_ds=test_ds, seed=seed)

            # ----run on unbinned data -----------------------

            if is_multiclass:
                data_synth_experiments(dta_name, binned=False, algs=mcc_algs, dt_ratios=dt_ratios, train_ds=train_ds,
                                       test_ds=test_ds, seed=seed)
            else:
                data_synth_experiments(dta_name, binned=False, algs=cont_algs, dt_ratios=dt_ratios, train_ds=train_ds,
                                       test_ds=test_ds, seed=seed)


def data_synth_experiments(
        dta_name,
        binned,
        algs,
        dt_ratios,
        train_ds,
        test_ds,
        seed=4711):
    if len(algs) == 0 or len(dt_ratios) == 0 or len(train_ds) == 0 or len(test_ds) == 0:
        return

    print(dta_name)
    X, y, N, Y, n_classes, y_cts, y_idx = helpers.get_xy(dta_name, load_from_disk=True, binned=binned)

    n_combs = len(dt_ratios) * len(train_ds) * len(test_ds)
    n_cols = 5 + 4 * n_classes + n_classes * len(algs)

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

                for str_alg in algs:
                    print(str_alg)

                    p = run_setup(str_alg, X, y, train_index, test_index)

                    print(p)

                    stats_matrix[i, j:(j + n_classes)] = p

                    j += n_classes

                i += 1

    col_names = ["Total_Samples_Used", "Training_Size", "Test_Size", "Training_Ratio", "Test_Ratio"]
    col_names += ["Training_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Training_Class_" + str(l) + "_Relative" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Absolute" for l in Y]
    col_names += ["Test_Class_" + str(l) + "_Relative" for l in Y]

    for alg in algs:
        for li in Y:
            col_names += [alg + "_Prediction_Class_" + str(li)]

    stats_data = pd.DataFrame(data=stats_matrix,
                              columns=col_names)

    fname = res_path + dta_name + "_seed_" + str(seed) + "_" + strftime("%Y-%m-%d_%H-%M-%S", localtime()) + ".csv"
    stats_data.to_csv(fname, index=False, sep=';')


def run_setup(str_alg,
              X,
              y,
              train_idx,
              test_idx,
              init_args=None,
              fit_args=None,
              pred_args=None):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    if init_args is None:
        init_args = []
    if fit_args is None:
        fit_args = [X_train, y_train]
    if pred_args is None:
        pred_args = [X_test]

    qf = algorithm_dict[str_alg](*init_args)

    qf.fit(*fit_args)
    p = qf.predict(*pred_args)

    return p


if __name__ == "__main__":
    args = parse_args()
    run_synth(args.datasets, args.algorithms, args.dt, args.mc, args.seeds)
