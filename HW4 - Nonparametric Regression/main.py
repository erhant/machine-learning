import csv
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from math import isnan

# Extracts dataset from csv file
def extractDataset(path):
    data = []
    lines = csv.reader(open(path, "r"))
    lines = list(lines)
    for line in lines[1:]:
        datum = line
        datumFloat = [float(feature) for feature in datum]
        data.append(datumFloat)
    return np.array(data)

# Root Mean Squared Error
def rmse(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) * (y_true - y_pred)) / len(y_true))

def minMax(arr):
    return (np.min(arr), np.max(arr))
    

def regressogram(X_train, X_test, y_train, bin_width, origin):
    X_min, X_max = minMax(X_train)
    X_min = origin # we are given the start point
    # We will define our window as parts of left and right borders
    bin_count = 0;
    i = X_min
    while (i < X_max): 
        i+=bin_width
        bin_count+=1
    left = np.arange(bin_count) * bin_width + origin
    right = left + bin_width

    predictions = np.zeros_like(X_test)
    means = np.zeros(bin_count)
    for i in range(bin_count):        
        means[i] = np.mean(y_train[np.logical_and(left[i] < X_train, X_train <= right[i])]) # only those who belong to this bin are selected
        predictions[np.logical_and(left[i] < X_test, X_test <= right[i])] = means[i]
    
    return predictions, means, left, right

def runningMeanSmoother(X_test, X_train, y_train, bin_width):
    predictions = np.zeros_like(X_test)
    for i in range(len(X_test)):
        x = X_test[i]
        mean = np.mean(y_train[np.logical_and((x - 0.5 * bin_width) < X_train, X_train < (x + 0.5 * bin_width))])
        if (isnan(mean)):
            mean = 0 # we set NaN's to be 0
        predictions[i] = mean
    return predictions

def kernelSmoother(X_test, X_train, y_train, bin_width):
    predictions = np.zeros_like(X_test)
    for i in range(len(X_test)):
        x = X_test[i]
        mean = np.sum(1 / np.sqrt(2 * np.pi) * np.exp((x - X_train) * (X_train - x) / (2 * kernel_bin_width * kernel_bin_width)) * (y_train)) / np.sum(1 / np.sqrt(2 * np.pi) * np.exp((x - X_train) * (X_train - x) / (2 * kernel_bin_width * kernel_bin_width)))
        if (isnan(mean)):
            mean = 0 # we set NaN's to be 0
        predictions[i] = mean
    return predictions
    
def plotRegressogram(left, right, means, plt):
    indices = np.append(left, right[-1])
    prints = np.zeros(((len(means)+1)*2, 2))
    for i in range(len(means)):
        prints[2*i,0] = indices[i]
        prints[2*i+1,0] = indices[i+1]
        prints[2*i,1] = means[i]
        prints[2*i+1,1] = means[i]
    prints[len(means)*2, 0] = indices[len(means)-1]
    prints[len(means)*2+1, 0] = indices[len(means)-1]
    prints[len(means)*2, 1] = means[len(means)-1]
    prints[len(means)*2+1, 1] = means[len(means)-1]
    plt.plot(prints[:,0], prints[:,1], color='black') # Plot is used here

def initPlot(X_train, X_test, y_train, y_test, bin_width):
    pplt = plt.figure().add_subplot(111)
    pplt.plot(X_train, y_train, 'o', color='blue');
    pplt.plot(X_test, y_test, 'o', color='red');
    pplt.set_ylabel('Waiting time to next eruption (min)')
    pplt.set_xlabel('Eruption time (min)')
    pplt.set_title('h = '+str(regressogram_bin_width))
    return pplt

###### START ######
# Initialize data
data = extractDataset('hw04_data_set.csv')
data_train = data[:150]
data_test = data[150:]
X_train = data_train[:,0]
y_train = data_train[:,1]
X_test = data_test[:,0]
y_test = data_test[:,1]

# Initialize parameters
regressogram_bin_width = 0.37 # same for all methods
rms_bin_width = 0.37 # same for all methods
kernel_bin_width = 0.37 # same for all methods
regressogram_origin = 1.5 # used by regressogram only

### Regressogram ###
# Plot the points first
regressogram_plt = initPlot(X_train, X_test, y_train, y_test, regressogram_bin_width)
# Calculate
predictions, means, left, right = regressogram(X_train, X_test, y_train, regressogram_bin_width, regressogram_origin)
error = rmse(y_test, predictions)
print("Regressogram => RMSE is ", error, " when h is ", regressogram_bin_width)
plotRegressogram(left, right, means, regressogram_plt)
plt.savefig('regressogram.png')

### Running Mean Smoother ###
# Plot the points first
rms_plot = initPlot(X_train, X_test, y_train, y_test, rms_bin_width)
# Calculate for test data
predictions = runningMeanSmoother(X_test, X_train, y_train, rms_bin_width)
error = rmse(y_test, predictions)
print("Running Mean Smoother => RMSE is ", error, " when h is ", rms_bin_width)
# Calculate the line for the graph
data_interval = np.linspace(start=1.5, stop=5.2, num=int((5.2-1.5)//0.01))
predictions = runningMeanSmoother(data_interval, X_train, y_train, rms_bin_width)
rms_plot.plot(data_interval, predictions, color='black')
plt.savefig('rms.png')

### Kernel Smoother ###
# Plot the points first
kernel_plot = initPlot(X_train, X_test, y_train, y_test, kernel_bin_width)
# Calculate for test data
predictions = kernelSmoother(X_test, X_train, y_train, kernel_bin_width)
error = rmse(y_test, predictions)
print("Kenrel Smoother => RMSE is ", error, " when h is ", kernel_bin_width)
# Calculate the line for the graph
data_interval = np.linspace(start=1.5, stop=5.2, num=int((5.2-1.5)//0.01))
predictions = kernelSmoother(data_interval, X_train, y_train, kernel_bin_width)
kernel_plot.plot(data_interval, predictions, color='black')
plt.savefig('kernel.png') 
####### END #######