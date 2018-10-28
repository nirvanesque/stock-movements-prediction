#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# --> Packages python 

# Necessary librarys
import os # it's a operational system library, to set some informations
import random # random is to generate random values

import pandas as pd # to manipulate data frames 
import numpy as np # to work with matrix
from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values

import matplotlib.pyplot as plt # to graphics plot
import seaborn as sns # a good library to graphic plots
#import squarify # to better understand proportion of categorys - it's a treemap layout algorithm

import json # to convert json in df
from pandas.io.json import json_normalize # to normalize the json file



columns = ['device', 'geoNetwork', 'totals', 'trafficSource'] # Columns that have json format

dir_path_train = "data/train.csv" # you can change to your local 
dir_path_test = "data/test.csv" # you can change to your local 


# p is a fractional number to skiprows and read just a random sample of the our dataset. 
p = 0.7 # *** In this case we will use 50% of data set *** #

#Code to transform the json format columns in table
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train = load_df(dir_path_train)
train.head(5)
train.columns

test = load_df(dir_path_test)
test.head(5)
test.columns