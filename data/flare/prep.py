
import pandas as pd


def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2"

    colnames = ["att" + str(i + 1) for i in range(10)] + ["C", "M", "X"]

    dta = pd.read_csv(url,
                      header=None,
                      names=colnames,
                      sep=' ',
                      skiprows=1,
                      skipinitialspace=True)

    dta = dta.drop(["M", "X", "att10"], axis=1)

    dta = pd.get_dummies(dta, columns=["att1", "att2", "att3", "att4", "att5", "att9", "att6"])
    dta.loc[dta['att7'] > 1, 'att7'] = 0
    dta.loc[dta['att8'] > 1, 'att8'] = 0

    bins = [-1, 0, 10]
    labels = [0, 1]
    dta['C'] = pd.cut(dta['C'], bins=bins, labels=labels)
    dta['C'] = dta['C'].astype("int64")

    return dta

# dta.to_pickle("dta.pkl")
