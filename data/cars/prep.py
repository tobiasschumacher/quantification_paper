# -*- coding: utf-8 -*-

import pandas as pd

def prep_data():

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

  colnames = ["buying",
          "maint",
          "doors",
          "persons",
          "lug_boot",
          "safety",
          "acc_class"]
  
  dta = pd.read_csv(url,
              names = colnames,
              index_col = False,
              skipinitialspace = True)
  
  dta.acc_class = dta.acc_class.replace({"unacc" : 0, 
                            "acc" : 1,
                            "good" : 2,
                            "vgood" : 3})
  
  dta = pd.get_dummies(dta)
 
  return dta

# dta.to_pickle("dta.pkl")
  