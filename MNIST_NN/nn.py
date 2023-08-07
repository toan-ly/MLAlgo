import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data.head()

data = np.array(data)
num_samples, num_features = data.shape
np.random.shuffle(data) # shuffle before splitting

data_dev = data[0:1000].T
X_dev = data_dev[1:num_features]
y_dev = data_dev[0]
X_dev = X_dev / 255

data_train = data[1000:num_samples].T
X_train = data_train[1:num_features]
y_train = data_train[0]
X_train = X_train / 255

def init_params():
  W1 = np.random.rand(10, 784) - 0.5
  b1 = np.random.rand(10, 1) - 0.5
  W2 = np.random.rand(10, 10) - 0.5
  b2 = np.random.rand(10, 1) - 0.5
  return W1, b1, W2, b2
  
def ReLU(z):
  return np.maximum(z, 0)

def ReLU_derivative(z):
  return z > 0

def softmax(z):
  return np.exp(z) / sum(np.exp(z))

def one_hot(y):
  onehot = np.zeros((y.size, y.max() + 1))
  onehot[np.arange(y.size), y] = 1
  onehot = onehot.T
  return onehot

def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, y):
  onehot_y = one_hot(y)
  dZ2 = A2 - onehot_y
  dW2 = 1 / num_samples * dZ2.dot(A1.T)
  db2 = 1 / num_samples * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
  dW1 = 1 / num_samples * dZ1.dot(X.T)
  db1 = 1 / num_samples * np.sum(dZ1)
  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
  W1 -= lr * dW1
  b1 -= lr * db1    
  W2 -= lr * dW2  
  b2 -= lr * db2    
  return W1, b1, W2, b2
  
def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(pred, y):
  print(pred, y)
  return np.sum(pred == y) / y.size

def gradient_descent(X, y, lr, epochs):
  W1, b1, W2, b2 = init_params()
  for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
    if epoch % 10 == 0:
      print("Epoch: ", epoch)
      predictions = get_predictions(A2)
      print(get_accuracy(predictions, y))
  return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
  _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
  return get_predictions(A2)

def test_prediction(index, W1, b1, W2, b2):
  current_image = X_train[:, index, None]
  prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
  label = y_train[index]
  print("Prediction: ", prediction)
  print("Label: ", label)
  
  current_image = current_image.reshape((28, 28)) * 255
  plt.gray()
  plt.imshow(current_image, interpolation='nearest')
  plt.show()