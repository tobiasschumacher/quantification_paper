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

  url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip")
  
  my_zip_file = ZipFile(BytesIO(url.read()))
  train_file = my_zip_file.namelist()[2]
  test_file1 = my_zip_file.namelist()[1]
  test_file2 = my_zip_file.namelist()[0]
  
  df_train =  pd.read_csv(my_zip_file.open(train_file),
                  header = 0,
                  skipinitialspace = True)
  
  df_test1 =  pd.read_csv(my_zip_file.open(test_file1),
                  header = 0,
                  skipinitialspace = True)
  
  df_test2 =  pd.read_csv(my_zip_file.open(test_file2),
                  header = 0,
                  skipinitialspace = True)
  
  dta = df_train.append(df_test1, ignore_index = True)
  dta = dta.append(df_test2, ignore_index = True)  

  dta = dta.drop("date", axis = 1)
  
  if binned:
    for col in list(dta)[:-1]:
      dta[col] = pd.qcut(dta[col], q = 4, labels = False, duplicates = 'drop')
      dta[col] = dta[col].astype("int64")
      
      
      # dta.to_pickle("dta_binned.pkl")

    
  return dta
  
# dta.to_pickle("dta.pkl")
