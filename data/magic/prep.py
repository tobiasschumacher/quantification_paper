
import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"

    colnames = ["fLength",
                "fWidth",
                "fSize",
                "fConc",
                "fConc1",
                "fAsym",
                "fM3Long",
                "fM3Trans",
                "fAlpha",
                "fDist",
                "target"]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta.target = dta.target.replace({"h": 0, "g": 1})

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
