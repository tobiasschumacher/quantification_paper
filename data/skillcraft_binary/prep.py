import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv"

    dta = pd.read_csv(url,
                      header=0,
                      skipinitialspace=True,
                      na_values="?")

    dta = dta.dropna()
    dta = dta.drop("GameID", axis=1)

    bins = [0, 4, 8]
    labels = [0, 1]
    dta['LeagueIndex'] = pd.cut(dta['LeagueIndex'], bins=bins, labels=labels)
    dta['LeagueIndex'] = dta['LeagueIndex'].astype("int64")

    if binned:
        for col in list(dta)[1:]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
