# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:44:56 2018

@author: tobi_
"""

from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned = False):

  url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip")
  
  my_zip_file = ZipFile(BytesIO(url.read()))
  train_file = my_zip_file.namelist()[1]
  test_file = my_zip_file.namelist()[2]
  
  colnames = ["F" + str(i+1) for i in range(10)] + ["Class"]
  
  df_train =  pd.read_csv(my_zip_file.open(train_file),
                  names = colnames,
                  skipinitialspace = True)
  
  df_test =  pd.read_csv(my_zip_file.open(test_file),
                  names = colnames,
                  skipinitialspace = True)
  
  dta = df_train.append(df_test, ignore_index = True)

  dta.loc[dta['Class'] != "A", 'Class'] = "B"
  dta.Class = dta.Class.replace({"A"  : 0, "B"  : 1})
  
  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
      
      # dta.to_pickle("dta_binned.pkl")

    
  return dta
  
# dta.to_pickle("dta.pkl")
