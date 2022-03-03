import csv
import numpy as np
import pandas as pd
import time
import datetime
from decimal import Decimal

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold  # K-FOLD CROSS VALIDATION
from sklearn.tree import DecisionTreeRegressor  ## TREE
from sklearn.neural_network import MLPRegressor  ## XGBOOST TREE
from sklearn.ensemble import RandomForestRegressor  ## RANDOM FOREST
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

sns.set()


# Write data to csv file
def exportDataset(data, path):
  return 0


# Extracts dataset from csv file
def extractDataset(path, dichotomy_year=True, dichotomy_trxtype=True):
  data = []
  atmIdDict = {}
  idCnt = 1
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines[1:]:  # dont skip the first line here
    datum = line
    datumFloat = [float(feature) for feature in datum[1:]]
    id = datum[0]
    if not id in atmIdDict:
      atmIdDict[id] = idCnt
      idCnt += 1
    datumId = atmIdDict[id]
    data.append([datumId] + datumFloat)
  data = np.array(data)
  if dichotomy_year:
    data[:, 4] -= 2018  # Translate years to 0, 1 (dichotomic)
  if dichotomy_trxtype:
    data[:, 5] -= 1  # Translate transaction type to 0, 1 (dichotomic)
  return data


# Calculate correlation
def displayCorrelation(X):
  # Use pandas for that
  df = pd.DataFrame(X)
  sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='Blues')


# Root Mean Squared Error
def rmse(y_true, y_pred):
  assert len(y_true) == len(y_pred)
  return np.sqrt(np.sum((y_true - y_pred) * (y_true - y_pred)) / len(y_true))


# Mean Absolute Error
def mae(y_true, y_pred):
  assert len(y_true) == len(y_pred)
  return np.sum(np.abs(y_true - y_pred)) / len(y_true)


def convertYMDtoSeconds(years, months, days):
  assert len(years) == len(months)
  assert len(months) == len(days)
  times = []
  epoch = datetime.datetime.utcfromtimestamp(0)
  for i in range(len(years)):
    dt = datetime.datetime(year=years[i], month=months[i], day=days[i])
    times.append((dt - epoch).total_seconds())
  return np.array(times)


def outputResults(name_of_method, fold_count, rmse_arr, mae_arr, elapsed):
  rmses = np.array(rmse_arr)
  maes = np.array(mae_arr)
  avg_elapsed = Decimal(np.mean(np.array(elapsed))).quantize(Decimal('1.00000'))
  avg_rmse = Decimal(np.mean(rmses)).quantize(Decimal('1.00000'))
  avg_mae = Decimal(np.mean(maes)).quantize(Decimal('1.00000'))
  min_rmse = Decimal(np.min(rmses)).quantize(Decimal('1.00000'))
  min_mae = Decimal(np.min(maes)).quantize(Decimal('1.00000'))
  max_rmse = Decimal(np.max(rmses)).quantize(Decimal('1.00000'))
  max_mae = Decimal(np.max(maes)).quantize(Decimal('1.00000'))
  print(name_of_method, "with", fold_count, "folds:")
  print("RMSE >>> Avg:", avg_rmse, "\tMin:", min_rmse, "\tMax:", max_rmse)
  print("MAE  >>> Avg:", avg_mae, "\tMin:", min_mae, "\tMax:", max_mae)
  print("Average training time (sec):", avg_elapsed)


def randomForestRegression(X_train, y_train, X_test, Kfold_split_count=5):
  kf = KFold(n_splits=Kfold_split_count)
  rmses = []
  maes = []
  elapsed = []
  for train, test in kf.split(X_train):
    # train and test are the indexes
    X_train_fold = X_train[train, :]
    y_train_fold = y_train[train]
    X_test_fold = X_train[test, :]
    y_test_fold = y_train[test]

    time_start = time.time()
    ########################
    regressor = RandomForestRegressor(max_depth=2, random_state=0)
    regressor.fit(X_train_fold, y_train_fold)
    y_test_fold_pred = regressor.predict(X_test_fold)
    ########################
    time_elapsed = time.time() - time_start

    y_test_fold_pred = regressor.predict(X_test_fold)

    rmse_fold = rmse(y_test_fold, y_test_fold_pred)
    mae_fold = mae(y_test_fold, y_test_fold_pred)
    rmses.append(rmse_fold)
    maes.append(mae_fold)
    elapsed.append(time_elapsed)
  outputResults("Random Forest Regression", Kfold_split_count, rmses, maes, elapsed)


def lightgbmRegression(X_train, y_train, X_test, Kfold_split_count=5):
  kf = KFold(n_splits=Kfold_split_count)
  rmses = []
  maes = []
  elapsed = []
  for train, test in kf.split(X_train):
    # train and test are the indexes
    X_train_fold = X_train[train, :]
    y_train_fold = y_train[train]
    X_test_fold = X_train[test, :]
    y_test_fold = y_train[test]

    time_start = time.time()
    ########################
    regressor = lgb.LGBMRegressor(num_leaves=34, learning_rate=0.2, boosting_type='gbdt')
    regressor.fit(X_train_fold, y_train_fold)
    ########################
    time_elapsed = time.time() - time_start

    y_test_fold_pred = regressor.predict(X_test_fold)

    rmse_fold = rmse(y_test_fold, y_test_fold_pred)
    mae_fold = mae(y_test_fold, y_test_fold_pred)
    rmses.append(rmse_fold)
    maes.append(mae_fold)
    elapsed.append(time_elapsed)
  outputResults("LightGBM Regression", Kfold_split_count, rmses, maes, elapsed)


def xgboostRegression(X_train, y_train, X_test, Kfold_split_count=5):
  kf = KFold(n_splits=Kfold_split_count)
  rmses = []
  maes = []
  elapsed = []
  for train, test in kf.split(X_train):
    # train and test are the indexes
    X_train_fold = X_train[train, :]
    y_train_fold = y_train[train]
    X_test_fold = X_train[test, :]
    y_test_fold = y_train[test]

    time_start = time.time()
    ########################
    regressor = xgb.XGBRegressor()
    regressor.fit(X_train_fold, y_train_fold)
    ########################
    time_elapsed = time.time() - time_start

    y_test_fold_pred = regressor.predict(X_test_fold)

    rmse_fold = rmse(y_test_fold, y_test_fold_pred)
    mae_fold = mae(y_test_fold, y_test_fold_pred)
    rmses.append(rmse_fold)
    maes.append(mae_fold)
    elapsed.append(time_elapsed)
  outputResults("XGBoost Regression", Kfold_split_count, rmses, maes, elapsed)


def decisionTreeRegression(X_train, y_train, X_test, Kfold_split_count=5, use_max_depth=0):
  kf = KFold(n_splits=Kfold_split_count)
  rmses = []
  maes = []
  elapsed = []
  for train, test in kf.split(X_train):
    # train and test are the indexes
    X_train_fold = X_train[train, :]
    y_train_fold = y_train[train]
    X_test_fold = X_train[test, :]
    y_test_fold = y_train[test]

    time_start = time.time()
    ########################
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train_fold, y_train_fold)
    if use_max_depth > 0:
      depth = regressor.tree_.max_depth  # decrease total depth
      regressor = DecisionTreeRegressor(random_state=0, max_depth=depth - use_max_depth)
      regressor.fit(X_train_fold, y_train_fold)
    ########################
    time_elapsed = time.time() - time_start

    y_test_fold_pred = regressor.predict(X_test_fold)

    rmse_fold = rmse(y_test_fold, y_test_fold_pred)
    mae_fold = mae(y_test_fold, y_test_fold_pred)
    rmses.append(rmse_fold)
    maes.append(mae_fold)
    elapsed.append(time_elapsed)
  outputResults("Decision Tree Regression", Kfold_split_count, rmses, maes, elapsed)
  # Now finally return the predictions for the test data after training with whole dataset
  #regressor = DecisionTreeRegressor(random_state=0)
  #regressor = regressor.fit(X_train, y_train)
  #depth = regressor.tree_.max_depth # decrease total depth
  #regressor = DecisionTreeRegressor(random_state=0, max_depth=depth-3)
  #regressor = regressor.fit(X_train, y_train)
  #y_test_pred = regressor.predict(X_test)
  #return y_test_pred


def multilayerPerceptronRegression(X_train, y_train, X_test, Kfold_split_count=5, options={}):
  if 'alpha' in options:
    alpha = options['alpha']
  else:
    alpha = 1e-3
  if 'hidden_layer_sizes' in options:
    hidden_layer_sizes = options['hidden_layer_sizes']
  else:
    hidden_layer_sizes = (50, 50)
  if 'learning_rate' in options:
    learning_rate = options['learning_rate']
  else:
    learning_rate = 1e-2

  kf = KFold(n_splits=Kfold_split_count)
  rmses = []
  maes = []
  elapsed = []
  for train, test in kf.split(X_train):
    # train and test are the indexes
    X_train_fold = X_train[train, :]
    y_train_fold = y_train[train]
    X_test_fold = X_train[test, :]
    y_test_fold = y_train[test]

    time_start = time.time()
    regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='relu',
                             solver='adam',
                             alpha=alpha,
                             batch_size='auto',
                             learning_rate='adaptive',
                             learning_rate_init=learning_rate,
                             max_iter=200)
    regressor = regressor.fit(X_train_fold, y_train_fold)
    time_elapsed = time.time() - time_start

    y_test_fold_pred = regressor.predict(X_test_fold)

    rmse_fold = rmse(y_test_fold, y_test_fold_pred)
    mae_fold = mae(y_test_fold, y_test_fold_pred)
    rmses.append(rmse_fold)
    maes.append(mae_fold)
    elapsed.append(time_elapsed)
  outputResults("Multilayer Perceptron Regression", Kfold_split_count, rmses, maes, elapsed)
  # Now finally return the predictions for the test data after training with whole dataset
  #regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', alpha=alpha, batch_size='auto', learning_rate='adaptive', learning_rate_init=learning_rate, max_iter = 200)
  #regressor = regressor.fit(X_train_fold, y_train_fold)
  #y_test_pred = regressor.predict(X_test)
  #return y_test_pred


'''
You are given two input data files, namely, training_data.csv and test_data.csv. The
training set contains 42,958 labeled data instances (47 ATMs x 457 days x 2 transaction
types), where each training data instance has 7 columns. IDENTITY column gives you
the unique identifier assigned to each ATM. REGION column shows the geographical
region of each ATM. DAY, MONTH, and YEAR columns give the transaction date.
TRX_TYPE column shows the transaction type (1: card present, 2: card not present).
TRX_COUNT is the number of cash withdrawals performed on the specified date.

Columns:
    0 - id
    1 - region
    2 - day
    3 - month
    4 - year
    5 - transaction type
    6 - output (transaction count)
'''
###### START ######
# Initialize data
data_train = extractDataset("training_data.csv", dichotomy_year=False)
X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = extractDataset("test_data.csv", dichotomy_year=False)

# Calculate correlation to have an idea on the features
displayCorrelation(data_train)
'''
id - output: -0.068
region - output: 0.1
day - output: -0.041
month - output: -0.0023 (very low!)
year - output: -0.0055 (very low!)
transaction type: -0.82

We can remove month and year as we see here.
'''

##########
# EXPERIMENTAL: Convert year month day to hours in a single column
#minutes = ((convertYMDtoSeconds(X_train[:,4].astype(int), X_train[:,3].astype(int), X_train[:,2].astype(int)) / 60 ) / 60 ) / 24
#X_train = np.concatenate((X_train[:,:2], minutes[:,np.newaxis], X_train[:,5:]), axis = 1)
# (performed slight worse)
##########

##########
# EXPERIMENTAL: Remove month and year because they have very small correlation to the output
#X_train = np.concatenate((X_train[:,:3], X_train[:,5:]), axis = 1)
##########

##########
# EXPERIMENTAL: Convert to dichotomous variables for region
regionColumn = X_train[:, 1]
lb = LabelBinarizer()
lb.fit(regionColumn)
regionOneHot = np.array(lb.transform(regionColumn))
X_train = np.concatenate((X_train[:, :1], regionOneHot, X_train[:, 2:]), axis=1)

regionColumn = X_test[:, 1]
lb = LabelBinarizer()
lb.fit(regionColumn)
regionOneHot = np.array(lb.transform(regionColumn))
X_test = np.concatenate((X_test[:, :1], regionOneHot, X_test[:, 2:]), axis=1)
##########

##########
# EXPERIMENTAL: We probably dont need ID of the ATM, as it would not matter to someone who withdraws cash.
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]
##########

# Machine Learning
## I have tried several methods, and I have chosen the one with smallest errors. More detail will be given in the report.

# DECISION TREE REGRESSION
'''
Decision Tree Regression with 5 folds:
RMSE >>> Avg: 25.36043  Min: 18.49399   Max: 33.21141
MAE  >>> Avg: 14.83110  Min: 11.15468   Max: 19.49199
Average training time (sec): 0.06453
'''
#decisionTreeRegression(X_train, y_train, X_test, 5, use_max_depth=2)

# MULTILAYER PERCEPTRON REGRESSION
'''
Multilayer Perceptron Regression with 5 folds:
RMSE >>> Avg: 26.71208  Min: 22.48353   Max: 33.65730
MAE  >>> Avg: 16.40309  Min: 13.44368   Max: 20.49677
Average training time (sec): 27.11760
(also hits the 200 iteration mark)
'''
#multilayerPerceptronRegression(X_train, y_train, X_test, 5)

# XGBOOST (Decision Tree based Gradient Booster)
'''
Decision Tree Regression with 5 folds:
RMSE >>> Avg: 25.70542  Min: 23.54053   Max: 28.86956
MAE  >>> Avg: 15.85994  Min: 14.29202   Max: 18.20748
Average training time (sec): 1.45104
'''
#xgboostRegression(X_train, y_train, X_test)

# RANDOM FOREST REGRESSOR
'''
Random Forest Regression with 5 folds:
RMSE >>> Avg: 24.93979  Min: 22.23739   Max: 27.05529
MAE  >>> Avg: 15.25840  Min: 13.20672   Max: 16.76424
Average training time (sec): 0.09117
'''
#randomForestRegression(X_train, y_train, X_test)

# lightGBM REGRESSOR
'''
LightGBM Regression with 5 folds:
RMSE >>> Avg: 25.92255  Min: 22.54532   Max: 31.69969
MAE  >>> Avg: 15.90920  Min: 13.74256   Max: 19.51387
Average training time (sec): 0.22333
'''
lightgbmRegression(X_train, y_train, X_test, Kfold_split_count=3)

### FINAL PREDICIONS ###
# I have decided to use lightgbm
print("\nTraining on lightGBM")
regressor = lgb.LGBMRegressor(num_leaves=34, learning_rate=0.2, boosting_type='gbdt')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Predictions made, writing to file...")
np.savetxt("test_predictions.csv", y_pred, delimiter=",")
