# -*- coding: utf-8 -*-
# --> Packages python 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
import scipy.stats as stats
from scipy.stats import skew
import matplotlib
import warnings
import scipy
import sklearn
import numpy
import json
import sys
import csv
import os
import datetime

# ----------
# --> Check package version
# ----------

print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

# -----------------------
# Load datasets 
# -----------------------

dataset = load_csv_file("/home/amoussoudjangban/workspace/popAnalytics/epopAuAnalytics.csv")

# Mising value 
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Delete row where missing value 
dataset.columns
dataset = dataset.drop(dataset.loc[dataset['advertisername'].isnull()].index)

# Type of columns 
dataset.dtypes

# Distinct Client 
# Unique type per categorial columns 
len(dataset['advertisername'].unique())

# Count distinct item for a categorial value 
dataset["advertisername"].value_counts()

# How many campaign per clients 
dataset[["advertisername","campaignid"]].groupby("advertisername").campaignid.nunique()
# Another ways
#dataset.groupby(by='advertisername', as_index=False).agg({'campaignid': pd.Series.nunique})

# Convert object to timestamp
# df.join(df['AB'].str.split('-', 1, expand=True).rename(columns={0:'A', 1:'B'})) 
dataset['startdayDate'] =  pd.to_datetime(dataset[['startday']].startday.str.split('T',1).str[0], format='%Y-%m-%d')
dataset['enddayDate'] =  pd.to_datetime(dataset[['enddate']].enddate.str.split('T',1).str[0], format='%Y-%m-%d')
dataset['CampaignIntervalTime'] = dataset['enddayDate'] - dataset['startdayDate']

# Filter for one client
dataset[dataset["advertisername"] == "Air NZ"]["status"]



