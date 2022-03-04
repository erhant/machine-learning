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
def initialV(path):
  V = []
  lines = csv.reader(open(path, "r"))
  lines = list(lines)
  for line in lines:
    vs = line
    vf = [float(v) for v in vs]
    V.append(vf)
  return V


# Sigmoid Function
def sigmoid(a):
  return (1 / (1 + np.exp(-a)))


# Softmax Function
def softmax(scores):
  scores = np.exp(scores)
  scores = scores / np.sum(scores, axis=1)[:, np.newaxis]  # row sum is axis=1
  return scores


# Safe logarithm function
def safelog(a):
  return np.log(a + 1e-100)


def cbindOnes(matrix):
  if (len(matrix.shape) == 1):
    return np.hstack((np.ones((matrix.shape[0], 1)), matrix[:, np.newaxis]))
  else:
    return np.hstack((np.ones((matrix.shape[0], 1)), matrix))


def rbindOnes(matrix):
  if (len(matrix.shape) == 1):
    return np.vstack((np.ones((1, matrix.shape[0])), matrix[np.newaxis, :]))
  else:
    return np.vstack((np.ones((1, matrix.shape[1])), matrix))


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


def gradientW(X, Z, V, y_true, y_pred):
  return -cbindOnes(X).T.dot((y_true - y_pred).dot(V[1:].T) * Z * (1 - Z))


def gradientV(Z, y_true, y_pred):
  return -Z.T.dot(y_true - y_pred)


def learn(X, y, W, V, eta, epsilon, max_iteration):
  iteration = 0

  Z = sigmoid(cbindOnes(X).dot(W))  # (500, 785).(785, 20) = (500, 20)
  y_pred = softmax(cbindOnes(Z).dot(V))  # (500, 21).(21, 5) = (500, 5)
  objective_values = [-np.sum(y * safelog(y_pred))]  # i = classes, t = data points

  while (1):
    Z = sigmoid(cbindOnes(X).dot(W))  # (500, 785).(785, 20) = (500, 20)
    y_pred = softmax(cbindOnes(Z).dot(V))  # (500, 21).(21, 5) = (500, 5)

    deltaV = gradientV(Z, y, y_pred)
    deltaW = gradientW(X, Z, V, y, y_pred)

    V[1:] = V[1:] - eta * deltaV
    W = W - eta * deltaW

    Z = sigmoid(cbindOnes(X).dot(W))
    y_pred = softmax(cbindOnes(Z).dot(V))

    objective_values.append(-np.sum(y * safelog(y_pred)))
    if iteration >= max_iteration or np.abs(objective_values[iteration] - objective_values[iteration + 1]) < epsilon:
      break

    iteration += 1

  return [y_pred, objective_values, W, V, iteration]


# H = 20
# X = N x D  with +1 being bias
# W = (D+1) x 20
# Z = X with bias . W = N x 20
# V = (20+1) x 5
# O = Z with bias . V = N x 5

#### START #####
# Initial data
images = extractImages('images.csv')  # A CSV file where each line holds pixels
labels = extractLabels('labels.csv')  # A CSV file where each line has a number for the label
labelToClass = {1: "T-Shirt", 2: "Trouser", 3: "Dress", 4: "Sneaker", 5: "Bag"}  # Labels
eta = 0.0005  # Initialize Learning rate to 0.0005
epsilon = 1e-3  # Initialize minimum error to 1e-3
max_iteration = 500  # Initialize maximum iteration count to 500
W = np.array(initialW('initial_W.csv'))  # Initial weights (I to H) (785, 20)
V = np.array(initialV('initial_V.csv'))  # Initial weight (H to O) (21, 5)

# Divide the data randomly into 2 parts: [:500] as training, [500:200] as test
X_train = np.array(images[:500])  # (500, 784)
y_train = np.array(labels[:500])  # (500, )
X_test = np.array(images[500:1000])  # (500, 784)
y_test = np.array(labels[500:1000])  # (500, )
N_train, C_train = X_train.shape[0], len(labelToClass)
N_test, C_test = X_test.shape[0], len(labelToClass)
y_train_oh = oneHotEncoding(y_train, N_train, C_train)

# Learn
y_train_pred, objective_values_train, W, V, iterationCount = learn(X_train, y_train_oh, W, V, eta, epsilon,
                                                                   max_iteration)
y_test_pred = softmax(cbindOnes(sigmoid(cbindOnes(X_test).dot(W))).dot(V))
print("Finished in this many iterations: ", iterationCount)

# Plot the Iterations / Error graph
fig, ax = plt.subplots()
ax.plot(np.arange(iterationCount + 2), objective_values_train)  # +2 one from the start and one before end
ax.set(xlabel='Iteration', ylabel='Error')
fig.savefig("test.png")
plt.show()

# Calculate confusion matrixes
y_train_pred_labels = labelPrediction(y_train_pred)
y_test_pred_labels = labelPrediction(y_test_pred)
conf_train = confusion_matrix(y_train_pred_labels, y_train, labels=[1, 2, 3, 4, 5])
print(conf_train)
print('\n')
conf_test = confusion_matrix(y_test_pred_labels, y_test, labels=[1, 2, 3, 4, 5])
print(conf_test)
