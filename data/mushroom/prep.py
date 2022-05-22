import pandas as pd


def prep_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    colnames = ['result',
                'ap-shape',
                'ap-surface',
                'ap-color',
                'ruises?',
                'dor',
                'ill-attachment',
                'ill-spacing',
                'ill-size',
                'ill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col=False,
                      skipinitialspace=True)

    dta.result = dta.result.replace({"p": 0, "e": 1})
    dta['ring-number'] = dta['ring-number'].replace({"n": 0, "o": 1, "t": 2})
    dta = dta.drop(columns=['stalk-root'])
    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")
