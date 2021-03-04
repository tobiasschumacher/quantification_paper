# -*- coding: utf-8 -*-

import pandas as pd

def prep_data(binned = False):

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

  dta = pd.read_excel(url,
                header = 1,
                index_col = "ID")

  dta = dta.rename(index=str, columns={"default payment next month": "DEFAULT_PAYMENT"})

  dta = pd.get_dummies(dta, columns = ['SEX', 'EDUCATION', 'MARRIAGE',])

  if binned:
    bins = [20, 30, 40, 50, 65, 90]
    labels = [1,2,3,4,5]
    dta['AGE'] = pd.cut(dta['AGE'], bins = bins, labels = labels)
    dta['AGE'] = dta['AGE'].astype("int64")

    bins = [-200000, 0, 5000, 25000, 50000, 100000, 500000, 1000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT1'] = pd.cut(dta['BILL_AMT1'], bins = bins, labels = labels)
    dta['BILL_AMT1'] = dta['BILL_AMT1'].astype("int64")
    
    bins = [-200000, 0, 5000, 25000, 50000, 100000, 500000, 1000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT2'] = pd.cut(dta['BILL_AMT2'], bins = bins, labels = labels)
    dta['BILL_AMT2'] = dta['BILL_AMT2'].astype("int64")
    
    bins = [-200000, 0, 5000, 25000, 50000, 100000, 500000, 2000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT3'] = pd.cut(dta['BILL_AMT3'], bins = bins, labels = labels)
    dta['BILL_AMT3'] = dta['BILL_AMT3'].astype("int64")
    
    bins = [-200000, 0, 5000, 25000, 50000, 100000, 500000, 2000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT4'] = pd.cut(dta['BILL_AMT4'], bins = bins, labels = labels)
    dta['BILL_AMT4'] = dta['BILL_AMT4'].astype("int64")
    
    bins = [-200000, 0, 5000, 25000, 50000, 100000, 500000, 2000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT5'] = pd.cut(dta['BILL_AMT5'], bins = bins, labels = labels)
    dta['BILL_AMT5'] = dta['BILL_AMT5'].astype("int64")
    
    bins = [-500000, 0, 5000, 25000, 50000, 100000, 500000, 2000000]
    labels = [-1,0,1,2,3,4,5]
    dta['BILL_AMT6'] = pd.cut(dta['BILL_AMT6'], bins = bins, labels = labels)
    dta['BILL_AMT6'] = dta['BILL_AMT6'].astype("int64")
    
    bins = [0, 25000, 50000, 100000, 200000, 500000, 2000000]
    labels = [1,2,3,4,5,6]
    dta['LIMIT_BAL'] = pd.cut(dta['LIMIT_BAL'], bins = bins, labels = labels)
    dta['LIMIT_BAL'] = dta['LIMIT_BAL'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT1'] = pd.cut(dta['PAY_AMT1'], bins = bins, labels = labels)
    dta['PAY_AMT1'] = dta['PAY_AMT1'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT2'] = pd.cut(dta['PAY_AMT2'], bins = bins, labels = labels)
    dta['PAY_AMT2'] = dta['PAY_AMT2'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT3'] = pd.cut(dta['PAY_AMT3'], bins = bins, labels = labels)
    dta['PAY_AMT3'] = dta['PAY_AMT3'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT4'] = pd.cut(dta['PAY_AMT4'], bins = bins, labels = labels)
    dta['PAY_AMT4'] = dta['PAY_AMT4'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT5'] = pd.cut(dta['PAY_AMT5'], bins = bins, labels = labels)
    dta['PAY_AMT5'] = dta['PAY_AMT5'].astype("int64")
    
    bins = [-1, 500, 1000, 2500, 5000, 10000, 10000000]
    labels = [1,2,3,4,5,6]
    dta['PAY_AMT6'] = pd.cut(dta['PAY_AMT6'], bins = bins, labels = labels)
    dta['PAY_AMT6'] = dta['PAY_AMT6'].astype("int64")
    
    # dta.to_pickle("dta_binned.pkl")

  return dta

#dta.to_pickle("dta.pkl")
