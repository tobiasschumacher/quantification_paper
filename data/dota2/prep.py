# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:44:56 2018

@author: tobi_
"""

from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data():

  url = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip")
  
  my_zip_file = ZipFile(BytesIO(url.read()))
  train_file = my_zip_file.namelist()[0]
  test_file = my_zip_file.namelist()[1]
  
  hero_cols = ["H" + str(i+1) for i in range(113)]
  colnames = ["Winner", "LocID", "GameMode", "GameType"] + hero_cols
  
  
  
  df_train =  pd.read_csv(my_zip_file.open(train_file),
                  names = colnames,
                  skipinitialspace = True)
  
  df_test =  pd.read_csv(my_zip_file.open(test_file),
                  names = colnames,
                  skipinitialspace = True)
  
  dta = df_train.append(df_test, ignore_index = True)

  dta = dta.drop(columns = "LocID")
  
  dta = pd.get_dummies(dta, columns = ["GameMode", "GameType"])
  
  return dta
  
# dta.to_pickle("dta.pkl")
