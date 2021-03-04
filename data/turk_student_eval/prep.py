# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:45:02 2018

@author: tobi_
"""

import pandas as pd

def prep_data():

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00262/turkiye-student-evaluation_generic.csv"

  dta = pd.read_csv(url,
              index_col = False,
              sep = ',',
              skipinitialspace = True)
  
  dta = dta.drop(columns = ['class','nb.repeat'])
  
  #dta.to_pickle("dta.pkl")
  
  return dta