import csv
import numpy as np

# from sklearn.metrics import confusion_matrix # you can use this for testing


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


# Normal distribution calculation
def normalDist(x, mean, stddev):
  return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(((mean - x) * (x - mean)) / (2 * stddev * stddev))


# Naive-Bayes implementation
def bayes(X, mean, stddev, prior):
  P = normalDist(X, mean, stddev)
  P = np.log(P)  # for computational reasons we take the log (it actually gave me inf for most values without log!)
  return np.sum(P) + np.log(prior)


#### START #####
# Read data
images = extractImages('images.csv')  # A CSV file where each line holds pixels

# Read labels
labels = extractLabels('labels.csv')  # A CSV file where each line has a number (1 or 2) for class.
labelToGender = {
    1: "Female",
    2: "Male"
}  # 1 is for Woman, 2 for Man, in total there are 180 Men 20 Woman, for both training and test.

# Divide the data randomly into 2 parts: [:100] as training, [100:200] as test
X_train = np.array(images[:200])
y_train = np.array(labels[:200])
X_test = np.array(images[200:400])
y_test = np.array(labels[200:400])

# Get the classes for easier calculations
X_train_c1 = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] == 1])
X_train_c2 = np.array([X_train[i] for i in range(len(y_train)) if y_train[i] == 2])
y_train_c1 = np.array(y_train[y_train == 1])
y_train_c2 = np.array(y_train[y_train == 2])
# Each image has 4096 pixels (64x64), estimate the means and stddevs
class1_prior = len(y_train_c1) / len(y_train)
class2_prior = len(y_train_c2) / len(y_train)

# Calculate the mean by summing along the column and divide by element count
class1_means = (np.sum(X_train_c1, axis=0) / len(y_train_c1))
class2_means = (np.sum(X_train_c2, axis=0) / len(y_train_c2))

# Recall that (a-b)^2 is a^2 - 2ab + b^2, we sum each one of the three terms on their own and then combine together
class1_stddev = np.sqrt((np.sum(X_train_c1 * X_train_c1, axis=0) - 2 * np.sum(
    np.dot(X_train_c1,
           np.dot(class1_means[:, np.newaxis], np.ones((1, len(class1_means)))) * np.identity(len(class1_means))),
    axis=0) + (len(y_train_c1) * (class1_means * class1_means))) / len(y_train_c1))
class2_stddev = np.sqrt((np.sum(X_train_c2 * X_train_c2, axis=0) - 2 * np.sum(
    np.dot(X_train_c2,
           np.dot(class2_means[:, np.newaxis], np.ones((1, len(class2_means)))) * np.identity(len(class2_means))),
    axis=0) + (len(y_train_c2) * (class2_means * class2_means))) / len(y_train_c2))
# NOTE: np.dot(class1_means[:,np.newaxis], np.ones((1, len(class1_means))))
# this code can be made faster, by creating a zeros matrix M, and then for the vector V that we dot product with [1, 1, 1, ..]
# we can just do M[:,:] = V[:]
#> print(means[,1])
#[1] 0.3796078 0.3982353 ...
#> print(means[,2])
#[1] 0.3902179 0.3944227 ...
#> print(deviations[,1])
#[1] 0.1442828 0.1488465 ...
#> print(deviations[,2])
#[1] 0.1705677 0.1728641 ...
#> print(priors)
#[1] 0.1 0.9

# Do the Naive-bayes Classification
# Basically speaking, we calculate the possibilities P(y=1|x_i) and P(y=2|x_i) and decide the class based on their scores.
# Recall the bayesian formula.
##
## P(y=c|x_i) = (P(x_i|y=c) * P(y=c))/P(x_i)
##
# We call this naive because we assume an independence among data and we wont really care about P(x_i) as we dont need to!
# Note that instead of -(x - mean)^2 we wrote (x - mean)(mean - x)
y_train_pred = np.empty_like(y_train)
y_test_pred = np.empty_like(y_train)
for i in range(len(y_train)):
  y_train_pred[i] = np.argmax(
      np.array([
          bayes(X_train[i], class1_means, class1_stddev, class1_prior),
          bayes(X_train[i], class2_means, class2_stddev, class2_prior)
      ]))

for i in range(len(y_test)):
  y_test_pred[i] = np.argmax(
      np.array([
          bayes(X_test[i], class1_means, class1_stddev, class1_prior),
          bayes(X_test[i], class2_means, class2_stddev, class2_prior)
      ]))

y_train_pred += 1  # shift 0,1 to 1,2 for correct class labelling
y_test_pred += 1  # shift 0,1 to 1,2 for correct class labelling

two = np.array([2])  # [2]
one = np.array([1])  # [1]
# Calculate the confusion matrix for training set
#  train  predict 1 predict 2
#  actual 1 18      2
#  actual 2 24      156
#conf_train = confusion_matrix(y_train, y_train_pred, labels=[1, 2]) # sklearn function for testing
conf_train = np.zeros((2, 2), dtype=int)
conf_train[1, 1] = np.sum(np.logical_and(y_train - one, y_train_pred - one))
conf_train[1, 0] = np.sum(np.logical_and(y_train - one, two - y_train_pred))
conf_train[0, 1] = np.sum(np.logical_and(two - y_train, y_train_pred - one))
conf_train[0, 0] = np.sum(np.logical_and(two - y_train, two - y_train_pred))

# Calculate the confusion matrix for test set
#  test    predict 1 predict 2
#  actual 1 15      5
#  actual 2 19      161
#conf_test = confusion_matrix(y_test, y_test_pred, labels=[1, 2]) # sklearn function for testing
conf_test = np.zeros((2, 2), dtype=int)
conf_test[1, 1] = np.sum(np.logical_and(y_test - one, y_test_pred - one))
conf_test[1, 0] = np.sum(np.logical_and(y_test - one, two - y_test_pred))
conf_test[0, 1] = np.sum(np.logical_and(two - y_test, y_test_pred - one))
conf_test[0, 0] = np.sum(np.logical_and(two - y_test, two - y_test_pred))

print("Confusion matrix for Training data:\n", conf_train)
print("Confusion matrix for Test data:\n", conf_test)