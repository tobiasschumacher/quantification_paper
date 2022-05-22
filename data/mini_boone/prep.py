
import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt"

    colnames = ["att" + str(i + 1) for i in range(50)]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      skiprows=1,
                      sep=' ',
                      na_values="-0.999000E+03",
                      skipinitialspace=True)

    dta["signal"] = 0
    dta.iloc[:36499, -1] = 1
    dta = dta.dropna()

    if binned:
        for col in list(dta)[:-1]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
