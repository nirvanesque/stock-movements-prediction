# -*- coding: utf-8 -*-

# ------------------------------------------------------------------
#  Author : Baruch AMOUSSOU-DJANGBAN
#           Data Scientist @ JCDecaux Corporate 
# ------------------------------------------------------------------

"""
This code implement all most linear algorithm for regression problem
"""
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


# Definie validation function 
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# 1. Lasso 
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Line by line
clf = Lasso(alpha =0.0005, random_state=1)
prediction = clf.fit(train, y_train).predict(test)

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = prediction
sub.to_csv('/home/amoussoudjangban/workspace/DataKaggle/HousePrices/submission.csv',index=False)


# 2. ElasticNet 
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# 3. Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))










