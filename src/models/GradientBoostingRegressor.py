# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
#  Author : Baruch AMOUSSOU-DJANGBAN
#           Data Scientist @ JCDecaux Corporate 
# ------------------------------------------------------------------

"""
This code implement all most linear algorithm for regression problem
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)





score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# Line by line
prediction = GBoost.fit(train, y_train).predict(test)

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = prediction
sub.to_csv('/home/amoussoudjangban/workspace/DataKaggle/HousePrices/submission.csv',index=False)

