import numpy as np
import pandas as pd
import time
from decimal import Decimal
from sklearn.metrics import roc_auc_score # AUROC score
from sklearn.model_selection import KFold # K-FOLD CROSS VALIDATION
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler # STANDARDIZER
from sklearn.decomposition import PCA # PRINCIPAL COMPONENT ANALYSIS
# Classifiers
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


'''
We train the classifier using those that are not NaN label. We run the test on all of them.
'''

# Some evaluations
def outputResults(name_of_method, fold_count, auroc_arr, elapsed_arr):
    aurocs = np.array(auroc_arr)
    elapsed = np.array(elapsed_arr)
    avg_elapsed = Decimal(np.mean(np.array(elapsed))).quantize(Decimal('1.00000'))
    avg_auroc = Decimal(np.mean(aurocs)).quantize(Decimal('1.00000'))
    min_auroc = Decimal(np.min(aurocs)).quantize(Decimal('1.00000'))
    max_auroc = Decimal(np.max(aurocs)).quantize(Decimal('1.00000'))
    print(name_of_method, "with",fold_count,"folds:")
    print("AUROC >>> Avg:",avg_auroc,"\tMin:", min_auroc, "\tMax:",max_auroc)
    print("Average training time (sec):",avg_elapsed)
    print("\n\n")
    
# Root Mean Squared Error
def rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.sum((y_true - y_pred) * (y_true - y_pred)) / len(y_true))

# Remove given features from dataframe
def removeFeatures(dataframe, features, verbose=False):
    for f in features:
        if verbose:
            print("Removing:",f)
        dataframe = dataframe.drop(f, axis=1)
    if verbose:
        print("Removed",len(features),"features because their correlations were below the set threshold")
    return dataframe
    
# Remove low correlated features
def removeLowCorrelations(data, label, threshold = 0.01, verbose = False):
    if verbose:
        print("Features with correlations below absolute",threshold,"will be dropped.")
    dataset = pd.concat([data, label], axis = 1)
    corr =  dataset.corr()
    targetCorr = pd.DataFrame(corr.iloc[:,-1][:-1].abs().sort_values())
    if verbose:
        print('TARGET CORR COLUMNS',targetCorr.columns.values)
    targetCorrFiltered = targetCorr[targetCorr.iloc[:,0] < threshold]
    featuresToRemove = targetCorrFiltered.index.values
    data = removeFeatures(data, featuresToRemove, verbose=verbose)
    return data, featuresToRemove

# Converts categorical data to dichotomous
def convertCategoricalToDichotomous(train, test, verbose = False):
    # to get the same columns for both train and test, we concate them together and do this.
    splitIndex = train.shape[0]
    #print("SHAPES:",train.shape,test.shape)
    data = pd.concat([train, test], axis = 0)
    categorical = data.select_dtypes(exclude=['int', 'float64', 'bool', 'number'])
    for f in categorical.columns:
        dichotomous = pd.get_dummies(categorical[f])
        data = data.drop(f, axis = 1)
        data = pd.concat([data, dichotomous], axis = 1) 
        if verbose:
            print("Column:",f,"has been converted to dichotomous, which resulted in",dichotomous.shape[1],"columns.")
    assert len(data.select_dtypes(exclude=['int', 'float64', 'bool', 'number']).columns) == 0 # to make sure
    return data.iloc[:splitIndex,:], data.iloc[splitIndex:,:]
        
# Output CSV files
def outputCSV(predictions):
    predictions.to_csv('hw08_test_predictions.csv', index=False)

# Preprocessing is done here
def preprocess(X_train, X_test, y_train, removeId=True, fillNans=True, removeLowCorrelateds=True, convertCategoryToBinary=True,  pca=True):
    # We dont need ID column usually
    if removeId:
        X_train = X_train.iloc[:,1:]
        X_test = X_test.iloc[:,1:]
        y_train = y_train.iloc[:,1:]
        
    # Fill NaN's in data
    if fillNans:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    
    # Look at the lowest correlations, remove those that are correlated below an absolute value threshold
    if removeLowCorrelateds:
        X_train, removedFeatures = removeLowCorrelations(X_train, y_train, threshold=0.005, verbose = False)
        X_test = removeFeatures(X_test, removedFeatures, verbose = False)
        
    # Convert categoricals to dichtomous
    if convertCategoryToBinary:
        X_train, X_test = convertCategoricalToDichotomous(X_train, X_test, verbose = False)
    
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
        pca = PCA(.95) # Retain 95% of the variance
        pca.fit(X_train_np_standard)
        X_train_np = pca.transform(X_train_np_standard)
        X_test_np = pca.transform(X_test_np_standard)
    
    return X_train_np, X_test_np, np.squeeze(y_train_np)

### Classifiers ###
# Light GBM
def LGB(train, target, test):
    classifier = lgb.LGBMClassifier(num_leaves=32,boosting_type='gbdt')
    classifier.fit(train, target)
    print("LGB Score:",classifier.score(train, target))
    prediction = classifier.predict_proba(test)[:,1]
    return prediction

# XGBoost
def XGB(train, target, test, rf = True):
    if rf:
        prtstr = "XGBRF Score"
        classifier = xgb.XGBClassifier()
    else:
        prtstr = "XGB Score"
        classifier = xgb.XGBRFClassifier()
    classifier.fit(train, target)
    print(prtstr,classifier.score(train, target))
    prediction = classifier.predict_proba(test)[:,1]
    return prediction

# Gradient Boosting Classifier
def GBC(train, target, test):
    classifier = GradientBoostingClassifier()
    classifier.fit(train, target)
    print("GBC Score:",classifier.score(train, target))
    prediction = classifier.predict_proba(test)[:,1]
    return prediction

# Gaussian Naive Bayes (BAD RESULT)
def GNB(train, target, test):
    classifier = GaussianNB()
    classifier.fit(train, target)
    print("GNB Score:",classifier.score(train, target))
    prediction = classifier.predict_proba(test)[:,1]
    return prediction


# Support Vector Machines (TAKES TOO MUCH TIME!!!)
def SVM(train, target, test):
    classifier = svm.SVC(gamma='scale', probability=True, kernel='rbf') # rbf best
    classifier.fit(train, target)
    print("SVM Score:",classifier.score(train, target))
    prediction = classifier.predict_proba(test)[:,1]
    return prediction
###################
    
# Read
X_train = pd.read_csv('hw08_training_data.csv')
y_train = pd.read_csv('hw08_training_label.csv')
X_test = pd.read_csv('hw08_test_data.csv')

# Do each problem differently, with respect to the label
columns = y_train.columns.values
id_df = y_train[columns[0]]
columns = columns[1:]

auroc_scores_on_training = []
predictionsDf = pd.DataFrame(index=X_test.index, columns=y_train.columns)
predictionsDf.loc[:,'ID'] = X_test.iloc[:,0]
for c in columns:
    df_label = pd.concat([id_df, y_train[c]], axis=1).dropna()
    df_train = X_train.loc[df_label.index, :]
    df_test = X_test
    # Preprocess
    np_train, np_test, np_label =  preprocess(df_train, df_test, df_label)
    #print("PREPROCESS DONE...\n")
    
    # Cross validation
    foldCount = 3
    kf = StratifiedKFold(n_splits=foldCount)
    auroc_scores = []
    elapsed_times = []
    for train, test in kf.split(np_train, np_label):
        fold_train = np_train[train,:]
        fold_train_label = np.squeeze(np_label[train])
        fold_test = np_train[test,:]
        fold_test_label = np.squeeze(np_label[test])
        
        time_start = time.time() 
        
        # Classification
        method = 'XGBRF'
        if method == 'LGB':
           fold_test_pred = LGB(fold_train, fold_train_label, fold_test)
        if method == 'XGB':
            fold_test_pred = XGB(fold_train, fold_train_label, fold_test, rf = False)
        if method == 'XGBRF':
            fold_test_pred = XGB(fold_train, fold_train_label, fold_test, rf = True)
        if method == 'GBC':
            fold_test_pred = GBC(fold_train, fold_train_label, fold_test)
            
        time_elapsed = time.time() - time_start
        elapsed_times.append(time_elapsed)
        
        # Evaluation
        '''
        We use auroc, a 0.5 means it is performing poorly with almost random choice
        We want high auroc score (closer to 1)
        '''
        auroc_score = roc_auc_score(fold_test_label, fold_test_pred)
        auroc_scores.append(auroc_score)
    outputResults("RESULT ("+c+")", foldCount, auroc_scores, elapsed_times)
    # I chose XGB with RF for the actual predictions
    #On training data
    np_train_pred = XGB(np_train, np_label, np_train, rf = True)
    auroc_score = roc_auc_score(np_label, np_train_pred)
    auroc_scores_on_training.append(auroc_score)
    #On test data
    np_test_pred = XGB(np_train, np_label, np_test, rf = True)
    predictionsDf.loc[:, c] = np_test_pred

# output predictions
outputCSV(predictionsDf)

    
    


