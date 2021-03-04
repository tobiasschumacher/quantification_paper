# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "voice.csv"

  dta = pd.read_csv(url,
              header = 0,
              skipinitialspace = True)
  
  dta.label = dta.label.replace({"male": 0, "female": 1})
  
  if binned:
    for col in list(dta[:-1]):
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
    
    #dta.to_pickle("dta_binned.pkl")
  
  return dta

#dta.to_pickle("dta.pkl")
