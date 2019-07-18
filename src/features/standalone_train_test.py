import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

INPUT_DIR = '../../data/raw/ieee-fraud-detection/'

TRAIN_TRANSACTION = 'train_transaction.csv'
TEST_TRANSACTION = 'test_transaction.csv'

TRAIN_IDENTITY = 'train_identity.csv'
TEST_IDENTITY = 'test_identity.csv'

OUTPUT_DIR = '../../data/processed/'

print ('reading train and test files')

train_transaction = pd.read_csv(INPUT_DIR + TRAIN_TRANSACTION, index_col='TransactionID')
test_transaction = pd.read_csv(INPUT_DIR +  TEST_TRANSACTION, index_col='TransactionID')

train_identity = pd.read_csv(INPUT_DIR +  TRAIN_IDENTITY, index_col='TransactionID')
test_identity = pd.read_csv(INPUT_DIR +  TRAIN_IDENTITY, index_col='TransactionID')


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print ('Train and Test Shape')
print (train.shape)
print (test.shape)

train.to_csv(OUTPUT_DIR + 'train.csv')
test.to_csv(OUTPUT_DIR + 'test.csv')

train = train.drop('isFraud', axis=1)
train = train.fillna(-999)
test = test.fillna(-999)

object_columns = []

for f in train.columns:
    if train[f].dtype == 'object' or test[f].dtype == 'object':
        object_columns.append(f)

np.save(OUTPUT_DIR + 'object_columns', object_columns)

