
import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"

    dta = pd.read_csv(url,
                      header=0,
                      skipinitialspace=True)

    dta = dta.drop(['p1', 'stab'], axis=1)
    dta.stabf = dta.stabf.replace({"unstable": 0, "stable": 1})

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
