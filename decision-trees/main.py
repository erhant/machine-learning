import csv
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

###
# Book: ch.9.2.2 p.220
###


# Class for a Node in tree, if is_terminal is true then this Node is a Terminal Node (Leaf)
class Node:
  is_terminal = None
  left_node = None
  right_node = None
  value = None

  def __init__(self, reaching_x, reaching_y, P):
    #print("I got: ",len(reaching_x))
    # Split
    x_sorted = np.sort(np.unique(reaching_x))
    split_candidate = np.empty((len(x_sorted) - 1))
    if len(split_candidate) == 0:
      self.is_terminal = True
      self.value = np.mean(reaching_y)
    else:
      for i in range(len(split_candidate)):
        split_candidate[i] = (x_sorted[i] + x_sorted[i + 1]) / 2  # middle points
      # Try all of them
      split_errors = np.zeros_like(split_candidate)
      for i in range(len(split_candidate)):
        s = split_candidate[i]
        left_y = reaching_y[reaching_x < s]
        right_y = reaching_y[reaching_x >= s]
        # Calculate error
        split_errors[i] = (np.sum((left_y - np.mean(left_y)) * (left_y - np.mean(left_y))) + np.sum(
            (right_y - np.mean(right_y)) * (right_y - np.mean(right_y)))) / len(reaching_y)

      # Get the best split
      best_split = split_candidate[np.argmin(split_errors)]

      # Split the data
      left_x = reaching_x[reaching_x < best_split]
      right_x = reaching_x[reaching_x >= best_split]
      left_y = reaching_y[reaching_x < best_split]
      right_y = reaching_y[reaching_x >= best_split]

      #print("Length of left x'es:",len(left_x))
      #print("Length of right x'es:",len(right_x))
      #print("\n\n")
      if len(left_x) == 0:
        # Only right
        self.is_terminal = True
        self.value = np.mean(right_y)
      elif len(right_x) == 0:
        # Only left
        self.is_terminal = True
        self.value = np.mean(left_y)
      elif len(left_x) + len(right_x) <= P:
        # Prune
        self.is_terminal = True
        self.value = np.mean(np.append(left_y, right_y))
      else:
        # This is not a terminal node!
        self.is_terminal = False
        # Create left and right nodes.
        self.value = best_split
        self.left_node = Node(left_x, left_y, P)
        self.right_node = Node(right_x, right_y, P)


# Class for decision tree
class DecisionTree:
  root = None

  def __init__(self, X, Y, P):
    self.root = Node(X, Y, P)

  def predict(self, X):
    node = self.root
    while not node.is_terminal:
      split_value = node.value
      if X <= split_value:
        node = node.left_node
      else:
        node = node.right_node
    return node.value


# Plot code for data
def initPlot(X_train, X_test, y_train, y_test):
  pplt = plt.figure().add_subplot(111)
  pplt.plot(X_train, y_train, 'o', color='blue')
  pplt.plot(X_test, y_test, 'o', color='red')
  pplt.set_ylabel('Waiting time to next eruption (min)')
  pplt.set_xlabel('Eruption time (min)')
  return pplt


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
  return Decimal(np.sqrt(np.sum((y_true - y_pred) * (y_true - y_pred)) / len(y_true))).quantize(Decimal('1.0000'))


# Cost function
def cost(y, mean):
  return np.sum((y - mean) * (y - mean))


###### START ######
# Initialize data
data = extractDataset('hw05_data_set.csv')
data_train = data[:150]
data_test = data[150:]
X_train = data_train[:, 0]
y_train = data_train[:, 1]
X_test = data_test[:, 0]
y_test = data_test[:, 1]

# Plot the points
dt_plt = initPlot(X_train, X_test, y_train, y_test)

# Construct the tree with P = 25
tree = DecisionTree(X_train, y_train, 25)
y_test_pred = [tree.predict(x) for x in X_test]
rmse_error = rmse(y_test, y_test_pred)
print("RMSE is", rmse_error, "when P is", 25)

# Draw the line
data_interval = np.linspace(start=1.5, stop=5.2, num=int((5.2 - 1.5) // 0.01))
data_interval_predict = [tree.predict(x) for x in data_interval]
dt_plt.plot(data_interval, data_interval_predict, color='black')
plt.savefig('dataplot.png')

# Try for all pruning parameters
PP = np.arange(5, 55, 5)
rmse_errors = np.array([])
for P in PP:
  tree = DecisionTree(X_train, y_train, P)
  y_test_pred = [tree.predict(x) for x in X_test]
  rmse_errors = np.append(rmse_errors, rmse(y_test, y_test_pred))

# Plot them
pp_rmse_plot = plt.figure().add_subplot(111)
pp_rmse_plot.scatter(PP, rmse_errors, color='black')
pp_rmse_plot.plot(PP, rmse_errors, color='black')
pp_rmse_plot.set_ylabel('RMSE')
pp_rmse_plot.set_xlabel('Pre-pruning size (P)')
plt.savefig('pptorms.png')
