import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.reduce_memory import reduce_mem_usage

INPUT_DIR = '../../data/processed/'

TRAIN = 'train.csv'
TEST = 'test.csv'

SAMPLE_SUBMISSION = 'sample_submission.csv'

OUTPUT_DIR = '../../models/'

print ("importing train and test files")

sample_submission = pd.read_csv('../../data/raw/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

train = pd.read_csv(INPUT_DIR + TRAIN)
test = pd.read_csv(INPUT_DIR + TEST)

train,NAlist  = reduce_mem_usage(train)
test ,NAlist  = reduce_mem_usage(test)

print(train.shape)
print(test.shape)

y = train['isFraud'].copy()

# Drop target, fill in NaNs
train = train.drop('isFraud', axis=1)

train = train.fillna(-999)
test = test.fillna(-999)

#
# test_ids = test['TransactionID']
#
# # identify constant features by looking at the standard deviation (check id std ==0.0)
# desc_train = train.describe().transpose()
# columns_to_drop = desc_train.loc[desc_train["std"] == 0].index.values
# train.drop(columns_to_drop, axis=1, inplace=True)
#
# # check which column has been dropped
# print("train columns to drop - " + columns_to_drop)
#
# desc_test = test.describe().transpose()
# columns_to_drop = desc_test.loc[desc_test["std"] == 0].index.values
# train.drop(columns_to_drop, axis=1, inplace=True)
#
# # check which column has been dropped
# print("train columns to drop - "+ columns_to_drop)


# process columns, apply LabelEncoder to categorical features
# for c in train.columns:
#     if train[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(train[c].values) + list(test[c].values))
#         train[c] = lbl.transform(list(train[c].values))
#         test[c] = lbl.transform(list(test[c].values))

# One-hot encoding of categorical/strings
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

train = train[:20000]
y=  y[:20000]

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)



clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    tree_method='hist'  # THE MAGICAL PARAMETER
)


clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

score = roc_auc_score(y_test,y_test_pred )
print   ("score is {} ".format(score))

del X_train, X_test, y_train, y_test

sample_submission['isFraud'] = clf.predict_proba(test)[:,1]
sample_submission.to_csv('xgboost_v1_{}.csv'.format(score))

print("done")





















# # One-hot encoding of categorical/strings
# train = pd.get_dummies(train, drop_first=True)
# test = pd.get_dummies(test, drop_first=True)
#
#
# train = train[:10000]
# # shape
# print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
# y_train = train["isFraud"]
# # y_mean = np.mean(y_train)
#
#
# #xgboost
# # prepare dict of params for xgboost to run with
# xgb_params = {
#     'n_trees': 500,
#     'eta': 0.005,
#     'max_depth': 4,
#     'subsample': 0.95,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     # 'base_score': y_mean, # base prediction = mean(target)
#     'silent': 1,
#     'colsample_bytree':0.4,
#     'n_estimators':1300
# }
#
# # form DMatrices for Xgboost training
# dtrain = xgb.DMatrix(train, y_train)
# dtest = xgb.DMatrix(test)
#
# # xgboost, cross-validation
# cv_result = xgb.cv(xgb_params,
#                    dtrain,
#                    num_boost_round=700, # increase to have better results (~700)
#                    early_stopping_rounds=50,
#                    verbose_eval=50,
#                    show_stdv=False
#                   )
#
# num_boost_rounds = len(cv_result)
# print(num_boost_rounds)
#
# # train model
# model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#
#
#
# # check f2-score (to get higher score - increase num_boost_round in previous cell)
# # now fixed, correct calculation
# roc_score  =roc_auc_score(dtrain.get_label(), model.predict(dtrain))
#
# print("scoe is {}".format(roc_score))
#
#
# # make predictions and save results
# y_pred = model.predict(dtest)
# output = pd.DataFrame({'TransactionID': test_ids, 'isFraud': y_pred})
# output.to_csv(OUTPUT_DIR+'xgboost_v1_{0}.csv'.format(roc_score), index=False)
#
# print("done")


#
#
# print  ("pre-processing")
# y_train = train['isFraud'].copy()
#
# # Drop target, fill in NaNs
# X_train = train.drop('isFraud', axis=1)
# X_test = test.copy()
# X_train = X_train.fillna(-999)
# X_test = X_test.fillna(-999)
#
# print ("lebel encoding")
# # Label Encoding
# for f in X_train.columns:
#     if X_train[f].dtype=='object' or X_test[f].dtype=='object':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(X_train[f].values) + list(X_test[f].values))
#         X_train[f] = lbl.transform(list(X_train[f].values))
#         X_test[f] = lbl.transform(list(X_test[f].values))
#
#
# print ("xgboost classifier")
# clf = xgb.XGBClassifier(n_estimators=500,
#                         n_jobs=4,
#                         max_depth=9,
#                         learning_rate=0.05,
#                         subsample=0.9,
#                         colsample_bytree=0.9,
#                         missing=-999)
#
# clf.fit(X_train, y_train)
#
# score = roc_auc_score(y_train,)
#
# print ("sample submission")
#
# sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
# sample_submission.to_csv(OUTPUT_DIR+ 'simple_xgboost_v1.csv')
#
# print ("done")
