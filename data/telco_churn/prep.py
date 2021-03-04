# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "churn.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta = dta.dropna()
  dta = dta.drop(["customerID"], axis = 1)
  dta.Churn = dta.Churn.replace({"No": 0, "Yes": 1})
  dta = pd.get_dummies(dta)
  
  
  if binned:
    for col in list(dta)[:4]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
