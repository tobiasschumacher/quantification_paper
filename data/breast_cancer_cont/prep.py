import pandas as pd


def prep_data(binned=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

    colnames = ["ID",
                "diagnosis",
                "C1_radius",
                "C1_texture",
                "C1_perimeter",
                "C1_area",
                "C1_smoothness",
                "C1_compactness",
                "C1_concavity",
                "C1_concave_points",
                "C1_symmetry",
                "C1_fractal_dimension",
                "C2_radius",
                "C2_texture",
                "C2_perimeter",
                "C2_area",
                "C2_smoothness",
                "C2_compactness",
                "C2_concavity",
                "C2_concave_points",
                "C2_symmetry",
                "C2_fractal_dimension",
                "C3_radius",
                "C3_texture",
                "C3_perimeter",
                "C3_area",
                "C3_smoothness",
                "C3_compactness",
                "C3_concavity",
                "C3_concave_points",
                "C3_symmetry",
                "C3_fractal_dimension"]

    dta = pd.read_csv(url,
                      names=colnames,
                      index_col="ID",
                      skipinitialspace=True)

    dta.diagnosis = dta.diagnosis.replace({"B": 0, "M": 1})

    if binned:
        for col in list(dta):
            if col == "diagnosis":
                continue
            dta[col] = pd.cut(dta[col], bins=5, labels=[1, 2, 3, 4, 5])
            dta[col] = dta[col].astype("int64")
            # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")
