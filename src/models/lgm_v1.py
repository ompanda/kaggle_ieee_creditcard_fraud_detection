# https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.reduce_memory import reduce_mem_usage

INPUT_DIR = '../../data/processed/'

TRAIN = 'train.csv'
TEST = 'test.csv'

SAMPLE_SUBMISSION = 'sample_submission.csv'

OUTPUT_DIR = '../../models/'

print ("importing train and test files")

# sample_submission = pd.read_csv('../../data/raw/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

train_df = pd.read_csv(INPUT_DIR + TRAIN)
# test_df = pd.read_csv(INPUT_DIR + TEST)


isFraud = 'isFraud'
predictors = train_df.columns.values.tolist()[2:]

print ("is fraud value distribution")
print (train_df.isFraud.value_counts())





# train_df,NAlist  = reduce_mem_usage(train_df)
# test_df ,NAlist  = reduce_mem_usage(test_df)
#
# print(train_df.shape)
# print(test_df.shape)
#
# y_train = train_df['isFraud'].copy()
#
# train_df = train_df.fillna(-999)
# test_df = test_df.fillna(-999)
#
# # Label Encoding
# for f in train_df.columns:
#     if (train_df[f].dtype=='object' or train_df[f].dtype=='object') and f!='isFraud':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(train_df[f].values) + list(test_df[f].values))
#         train_df[f] = lbl.transform(list(train_df[f].values))
#         test_df[f] = lbl.transform(list(test_df[f].values))
# train_df = train_df.reset_index()
# test_df = test_df.reset_index()
#
# features = list(train_df)
# features.remove('isFraud')
# target = 'isFraud'
#
#
# # Bounded region of parameter space
# bounds_LGB = {
#     'num_leaves': (5, 200),
#     'min_data_in_leaf': (5, 200),
#     'bagging_fraction' : (0.1,0.9),
#     'feature_fraction' : (0.1,0.9),
#     'learning_rate': (0.01, 0.3),
#     'min_child_weight': (0.00001, 0.01),
#     'min_child_samples':(100, 500),
#     'subsample': (0.2, 0.8),
#     'colsample_bytree': (0.4, 0.6),
#     'reg_alpha': (1, 2),
#     'reg_lambda': (1, 2),
#     'max_depth':(-1,15),
# }
#
# LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
#
# print(LGB_BO.space.keys)
#
# init_points = 30
# n_iter = 3
#
# print('-' * 130)
#
# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore')
#     LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
#
# print(LGB_BO.max['target'])
#
# print (LGB_BO.max['params'])
#
# param_lgb = {
#         'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
#         'num_leaves': int(LGB_BO.max['params']['num_leaves']),
#         'learning_rate': LGB_BO.max['params']['learning_rate'],
#         'min_child_weight': LGB_BO.max['params']['min_child_weight'],
#         'colsample_bytree' : LGB_BO.max['params']['colsample_bytree'],
#         'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
#         'min_child_samples': LGB_BO.max['params']['min_child_samples'],
#         'subsample': LGB_BO.max['params']['subsample'],
#         'reg_lambda': LGB_BO.max['params']['reg_lambda'],
#         'reg_alpha': LGB_BO.max['params']['reg_alpha'],
#         'max_depth': int(LGB_BO.max['params']['max_depth']),
#         'objective': 'binary',
#         'save_binary': True,
#         'seed': 1337,
#         'feature_fraction_seed': 1337,
#         'bagging_seed': 1337,
#         'drop_seed': 1337,
#         'data_random_seed': 1337,
#         'boosting_type': 'gbdt',
#         'verbose': 1,
#         'is_unbalance': False,
#         'boost_from_average': True,
#         'metric':'auc'
#     }
#
# sample_submission['isFraud'] = predictions
# sample_submission.to_csv('submission_IEEE.csv')

