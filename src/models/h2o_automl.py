# using h2o Auto ML
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')

INPUT_DIR = '../../data/processed/'

TRAIN = 'train.csv'
TEST = 'test.csv'


OUTPUT_DIR = '../../models/'

print ("importing train and test files")
train = h2o.import_file(INPUT_DIR + TRAIN)
test = h2o.import_file(INPUT_DIR + TEST)

train.head()
test.head()

x = train.columns[1:]
y = 'isFraud'

print  ("auto ml starting")

aml = H2OAutoML(max_models=10, seed=47, max_runtime_secs=10000)
aml.train(x=x, y=y, training_frame=train)

print  ("leaderboard")
# View the AutoML Leaderboard
lb = aml.leaderboard
print (lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

print ("leader model")
print (aml.leader)

print ("predicting")
preds = aml.predict(test)
print (preds['p1'].as_data_frame().values.flatten().shape)

print ("preparing submission file")
df_submission = pd.DataFrame()
df_submission['TransactionID'] = test['TransactionID']
df_submission[y] = preds['p1'].as_data_frame().values

print ("saving submission file")
df_submission.to_csv(OUTPUT_DIR + "automl_v1.csv")

print ("done")