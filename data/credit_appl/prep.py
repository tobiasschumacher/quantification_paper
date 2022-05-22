
import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

    dta = pd.read_csv(url,
                      names=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14",
                             "A15", "A16"],
                      na_values=['?'],
                      skipinitialspace=True)

    dta = dta.dropna()
    dta = dta.reset_index(drop=True)
    dta.A1 = dta.A1.replace({"b": 0, "a": 1})
    dta.A9 = dta.A9.replace({"f": 0, "t": 1})
    dta.A10 = dta.A10.replace({"f": 0, "t": 1})
    dta.A12 = dta.A12.replace({"f": 0, "t": 1})
    dta.A16 = dta.A16.replace({"-": 0, "+": 1})
    dta = pd.get_dummies(dta, columns=["A4", "A5", "A6", "A7", "A13"])

    if binned:
        bins = [10, 20, 25, 30, 35, 40, 50, 80]
        labels = [1, 2, 3, 4, 5, 6, 7]
        dta['A2'] = pd.cut(dta['A2'], bins=bins, labels=labels)
        dta['A2'] = dta['A2'].astype("int64")

        bins = [-0.1, 0.5, 1, 2, 3, 5, 10, 30]
        labels = [1, 2, 3, 4, 5, 6, 7]
        dta['A3'] = pd.cut(dta['A3'], bins=bins, labels=labels)
        dta['A3'] = dta['A3'].astype("int64")

        bins = [-0.1, 0.1, 0.5, 1, 2, 3, 5, 10, 30]
        labels = [1, 2, 3, 4, 5, 6, 7, 8]
        dta['A8'] = pd.cut(dta['A8'], bins=bins, labels=labels)
        dta['A8'] = dta['A8'].astype("int64")

        bins = [-0.1, 0, 5, 10, 100]
        labels = [0, 1, 2, 3]
        dta['A11'] = pd.cut(dta['A11'], bins=bins, labels=labels)
        dta['A11'] = dta['A11'].astype("int64")

        bins = [-0.1, 50, 100, 250, 500, 2000]
        labels = [1, 2, 3, 4, 5]
        dta['A14'] = pd.cut(dta['A14'], bins=bins, labels=labels)
        dta['A14'] = dta['A14'].astype("int64")

        bins = [-0.1, 0, 5, 100, 500, 100000]
        labels = [0, 1, 2, 3, 4]
        dta['A15'] = pd.cut(dta['A15'], bins=bins, labels=labels)
        dta['A15'] = dta['A15'].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
