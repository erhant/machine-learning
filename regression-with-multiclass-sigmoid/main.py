import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Extracts label data from csv file
def extractLabels(path):
  labels = []
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines:
    label = int(line[0])
    labels.append(label)
  return labels


# Extracts image data from csv file
def extractImages(path):
  images = []
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines:
    pixels = line
    pixelsFloat = [float(pixel) for pixel in pixels]
    images.append(pixelsFloat)
  return images


# Extracts label data from csv file
def initialW(path):
  W = []
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines:
    ws = line
    wf = [float(w) for w in ws]
    W.append(wf)
  return W


# Extracts label data from csv file
def initialW0(path):
  W0 = []
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines:
    b = float(line[0])
    W0.append(b)
  return W0


# Sigmoid Function
def sigmoid(X, W, W0):
  return (1 / (1 + np.exp(-(X.dot(W) + W0))))


# Gradient for the Weights
def gradientW(X, y_truth, y_pred):
  classCount = y_truth.shape[1]
  results = np.empty((X.shape[1], classCount))
  for c in range(classCount):
    results[:, c] = np.sum(((y_truth[:, c] - y_pred[:, c]) * y_pred[:, c] * (1 - y_pred[:, c]))[:, np.newaxis] * X,
                           axis=0)
  return -results
  ##return -np.sum((y_truth - y_predicted)[:,np.newaxis] * X, axis = 0)


# Gradient for the biases
def gradientW0(y_truth, y_pred):
  return -np.sum((y_truth - y_pred) * y_pred * (1 - y_pred), axis=0)


def oneHotEncoding(y, sampleCount, labelCount):
  # Input is for example [1, 2, 2, 5]
  # Output is then: 1 0 0 0 0
  #                 0 1 0 0 0
  #                 0 1 0 0 0
  #                 0 0 0 0 1
  y_oh = np.zeros((sampleCount, labelCount))
  y_oh[np.arange(sampleCount)[:], y[:] - 1] = 1
  return y_oh


def labelPrediction(y_pred):
  numSamples = y_pred.shape[0]
  y_labels = np.zeros(numSamples, dtype=int)
  for i in range(numSamples):
    y_labels[i] = np.argmax(y_pred[i]) + 1
  return y_labels


def regression(X, y, W, W0, eta, epsilon, max_iteration):
  N = X.shape[0]
  C = W.shape[1]
  y = oneHotEncoding(y, N, C)
  iteration = 0
  objective_values = []
  while (1):  # To enter or not to enter the loop
    y_pred = sigmoid(X, W, W0)

    objective_value = np.sum(0.5 * (y - y_pred) * ((y - y_pred)))
    objective_values.append(objective_value)

    oldW = W
    oldW0 = W0

    W = W - eta * gradientW(X_train, y, y_pred)
    W0 = W0 - eta * gradientW0(y, y_pred)

    changeInWs = np.sqrt(np.sum((W0 - oldW0) * (W0 - oldW0)) + np.sum((W - oldW) * (W - oldW)))
    if iteration >= max_iteration or changeInWs < epsilon:
      break

    iteration += 1
  return [y_pred, objective_values, W, W0]


#### START #####
# Initial data
images = extractImages('images.csv')  # A CSV file where each line holds pixels
labels = extractLabels('labels.csv')  # A CSV file where each line has a number for the label
labelToClass = {1: "T-Shirt", 2: "Trouser", 3: "Dress", 4: "Sneaker", 5: "Bag"}  # Labels
eta = 0.0001  # Initialize Learning rate to 0.0001
epsilon = 1e-3  # Initialize minimum error to 1e-3
max_iteration = 500  # Initialize maximum iteration count to 500
W = np.array(initialW('initial_W.csv'))  # Initial weights (784, 5)
W0 = np.array(initialW0('initial_W0.csv'))  # Initial biases (5, )

# Divide the data randomly into 2 parts: [:100] as training, [100:200] as test
X_train = np.array(images[:500])  # (500, 784)
y_train = np.array(labels[:500])  # (500, )
X_test = np.array(images[500:1000])  # (500, 784)
y_test = np.array(labels[500:1000])  # (500, )
N_train, D_train = X_train.shape[0], 5
N_test, D_test = X_test.shape[0], 5
# Multiclass Binary Regression
y_train_pred, objective_values_train, W, W0 = regression(X_train, y_train, W, W0, eta, epsilon, max_iteration)
y_test_pred = sigmoid(X_test, W, W0)  # Use the resulting W and W0 to predict test data

# Plot the Iterations / Error graph
fig, ax = plt.subplots()
ax.plot(np.arange(max_iteration + 1), objective_values_train)
ax.set(xlabel='Iteration', ylabel='Error')
fig.savefig("test.png")
plt.show()

# Calculate confusion matrixes
y_train_pred_labels = labelPrediction(y_train_pred)
y_test_pred_labels = labelPrediction(y_test_pred)
conf_train = confusion_matrix(y_train, y_train_pred_labels, labels=[1, 2, 3, 4, 5])
print(conf_train)
print('\n')
conf_test = confusion_matrix(y_test, y_test_pred_labels, labels=[1, 2, 3, 4, 5])
print(conf_test)