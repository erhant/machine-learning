import numpy as np
from decimal import Decimal
import time
import pandas as pd
from sklearn.metrics import roc_auc_score  # AUROC score
from sklearn.model_selection import KFold  # K-FOLD CROSS VALIDATION
from sklearn.model_selection import StratifiedKFold  # Stratified K-FOLD
from sklearn.preprocessing import StandardScaler  # STANDARDIZER
from sklearn.decomposition import PCA  # PRINCIPAL COMPONENT ANALYSIS
from sklearn.gaussian_process.kernels import RBF  # needed for GaussianProcess
# Classifiers
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
''' 
For the Xth problem, the files are named as hw07_targetX_training_data.csv,
hw07_targetX_training_label.csv, and hw07_targetX_test_data.csv.
You will train using the data and labels, and make predictions on test data.
You will output you predictions as hw07_targetX_test_predictions.csv
X \in \{1, 2, 3\}
'''


# Some evaluations
def outputResults(name_of_method, fold_count, auroc_arr, elapsed_arr):
  aurocs = np.array(auroc_arr)
  elapsed = np.array(elapsed_arr)
  avg_elapsed = Decimal(np.mean(np.array(elapsed))).quantize(Decimal('1.00000'))
  avg_auroc = Decimal(np.mean(aurocs)).quantize(Decimal('1.00000'))
  min_auroc = Decimal(np.min(aurocs)).quantize(Decimal('1.00000'))
  max_auroc = Decimal(np.max(aurocs)).quantize(Decimal('1.00000'))
  print(name_of_method, "with", fold_count, "folds:")
  print("AUROC >>> Avg:", avg_auroc, "\tMin:", min_auroc, "\tMax:", max_auroc)
  print("Average training time (sec):", avg_elapsed)
  print("\n\n")


# Root Mean Squared Error
def rmse(y_true, y_pred):
  assert len(y_true) == len(y_pred)
  return np.sqrt(np.sum((y_true - y_pred) * (y_true - y_pred)) / len(y_true))


# Remove given features from dataframe
def removeFeatures(dataframe, features, verbose=False):
  for f in features:
    if verbose:
      print("Removing:", f)
    dataframe = dataframe.drop(f, axis=1)
  if verbose:
    print("Removed", len(features), "features because their correlations were below the set threshold")
  return dataframe


# Remove low correlated features
def removeLowCorrelations(data, label, threshold=0.01):
  print("Features with correlations below absolute", threshold, "will be dropped.")
  dataset = pd.concat([data, label], axis=1)
  corr = dataset.corr()
  targetCorr = pd.DataFrame(corr.iloc[:, -1][:-1].abs().sort_values())
  targetCorrFiltered = targetCorr[targetCorr['TARGET'] < threshold]
  featuresToRemove = targetCorrFiltered.index.values
  data = removeFeatures(data, featuresToRemove, verbose=True)
  return data, featuresToRemove


# Converts categorical data to dichotomous
def convertCategoricalToDichotomous(train, test, verbose=False):
  # to get the same columns for both train and test, we concate them together and do this.
  splitIndex = train.shape[0]
  data = pd.concat([train, test], axis=0)
  categorical = data.select_dtypes(exclude=['int', 'float64', 'bool', 'number'])
  for f in categorical.columns:
    dichotomous = pd.get_dummies(categorical[f])
    data = data.drop(f, axis=1)
    data = pd.concat([data, dichotomous], axis=1)
    if verbose:
      print("Column:", f, "has been converted to dichotomous, which resulted in", dichotomous.shape[1], "columns.")
  assert len(data.select_dtypes(exclude=['int', 'float64', 'bool', 'number']).columns) == 0  # to make sure
  return data.iloc[:splitIndex, :], data.iloc[splitIndex:, :]


# Output CSV files
def outputCSV(predictions):
  for i in range(len(predictions)):
    predictions[i].to_csv('hw07_target' + str(i + 1) + '_test_predictions.csv', index=False)


# Preprocessing is done here
def preprocess(X_train,
               X_test,
               y_train,
               removeId=True,
               fillNans=True,
               removeLowCorrelateds=True,
               convertCategoryToBinary=True,
               pca=True):
  # We dont need ID column usually
  if removeId:
    X_train = X_train.iloc[:, 1:]
    X_test = X_test.iloc[:, 1:]
    y_train = y_train.iloc[:, 1:]

  # Fill NaN's in data
  if fillNans:
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

  # Look at the lowest correlations, remove those that are correlated below an absolute value threshold
  if removeLowCorrelateds:
    X_train, removedFeatures = removeLowCorrelations(X_train, y_train, threshold=0.005)
    X_test = removeFeatures(X_test, removedFeatures, verbose=False)

  # Convert categoricals to dichtomous
  if convertCategoryToBinary:
    X_train, X_test = convertCategoricalToDichotomous(X_train, X_test, verbose=True)

  # Convert to numpy arrays
  X_train_np = X_train.to_numpy().astype(dtype='float64')
  X_test_np = X_test.to_numpy().astype(dtype='float64')
  y_train_np = y_train.to_numpy().astype(dtype='float64')

  # Do PCA
  if pca:
    # Standardize
    X_test_np_standard = StandardScaler().fit_transform(X_test_np)
    X_train_np_standard = StandardScaler().fit_transform(X_train_np)

    # Dimensionality reduction using PCA
    pca = PCA(.95)  # Retain 95% of the variance
    pca.fit(X_train_np_standard)
    X_train_np = pca.transform(X_train_np_standard)
    X_test_np = pca.transform(X_test_np_standard)

  return X_train_np, X_test_np, y_train_np


### ALGORITHMS ###
# Support Vector Machines
def SVM(train, target, test):
  classifier = svm.SVC(gamma='scale', probability=True, kernel='rbf')  # rbf best
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Gaussian Process Classifier (DONT EVEN TRY THIS IT TAKES TOO MUCH TIME)
def GPC(train, target, test):
  kernel = 1.0 * RBF(1.0)
  gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
  gpc.fit(train, target)
  #print("Score:",gpc.score(train, target))
  prediction = gpc.predict_proba(test)[:, 1]
  return prediction


# Light GBM
def LGB(train, target, test):
  classifier = lgb.LGBMClassifier(num_leaves=32, boosting_type='gbdt')
  classifier.fit(train, target)
  print("Score:", classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Gaussian Naive Bayes
def GNB(train, target, test):
  classifier = GaussianNB()
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Gradient Boosting Classifier
def GBC(train, target, test):
  classifier = GradientBoostingClassifier()
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Multinomial Naive Bayes (DOES NOT ACCEPT NEGATIVE VALUES)
def MNB(train, target, test):
  classifier = MultinomialNB()
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Random Forest
def RNF(train, target, test):
  classifier = RandomForestClassifier(n_estimators=100)
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# Multilayer Perceptron
def MLP(train, target, test):
  classifier = MLPClassifier(alpha=0.0002, learning_rate='adaptive')
  classifier.fit(train, target)
  #print("Score:",classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


# XGBoost
def XGB(train, target, test, rf=True):
  if rf:
    prtstr = "XGBRF Score"
    classifier = xgb.XGBClassifier()
  else:
    prtstr = "XGB Score"
    classifier = xgb.XGBRFClassifier()
  classifier.fit(train, target)
  print(prtstr, classifier.score(train, target))
  prediction = classifier.predict_proba(test)[:, 1]
  return prediction


###


def run(path_train, path_target, path_test):

  # Read data
  X_train = pd.read_csv(path_train)
  y_train = pd.read_csv(path_target)
  X_test = pd.read_csv(path_test)

  test_ids = X_test['ID']
  # Preprocess
  X_train_np, X_test_np, y_train_np = preprocess(X_train, X_test, y_train, removeLowCorrelateds=True)
  print("PREPROCESS DONE...\n")

  # Cross validation
  foldCount = 3
  kf = StratifiedKFold(n_splits=foldCount)
  auroc_scores = []
  elapsed_times = []
  for train, test in kf.split(X_train_np, y_train_np):
    X_train_fold = X_train_np[train, :]
    y_train_fold = np.squeeze(y_train_np[train])
    X_test_fold = X_train_np[test, :]
    y_test_fold = np.squeeze(y_train_np[test])

    time_start = time.time()

    # Classification
    y_pred_fold = XGB(X_train_fold, y_train_fold, X_test_fold, rf=True)
    time_elapsed = time.time() - time_start
    elapsed_times.append(time_elapsed)
    # Evaluation
    '''
        We use auroc, a 0.5 means it is performing poorly with almost random choice
        We want high auroc score (closer to 1)
        '''
    auroc_score = roc_auc_score(y_test_fold, y_pred_fold)
    auroc_scores.append(auroc_score)
  outputResults("RESULT:", foldCount, auroc_scores, elapsed_times)
  y_train_np = np.squeeze(y_train_np)

  # I chose LGB for the actual predictions
  #On training data
  y_pred_np = XGB(X_train_np, y_train_np, X_train_np, rf=True)
  auroc_score = roc_auc_score(y_train_np, y_pred_np)
  #On test data
  y_pred_np = XGB(X_train_np, y_train_np, X_test_np, rf=True)
  return pd.concat([test_ids, pd.DataFrame(data=y_pred_np, columns=['TARGET'])], axis=1), auroc_score


target1_pred, target1_auroc = run('hw07_target1_training_data.csv', 'hw07_target1_training_label.csv',
                                  'hw07_target1_test_data.csv')
target2_pred, target2_auroc = run('hw07_target2_training_data.csv', 'hw07_target2_training_label.csv',
                                  'hw07_target2_test_data.csv')
target3_pred, target3_auroc = run('hw07_target3_training_data.csv', 'hw07_target3_training_label.csv',
                                  'hw07_target3_test_data.csv')

outputCSV([target1_pred, target2_pred, target3_pred])