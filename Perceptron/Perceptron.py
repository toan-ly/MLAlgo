import numpy as np

def unit_step_func(x):
  return np.where(x > 0, 1, 0)

class Perceptron:
  def __init__(self, lr=0.01, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.activation_func = unit_step_func
    self.weights = None
    self.bias = None
    
  def fit(self, X, y):
    n_samples, n_features = X.shape

    self.weights = np.zeros(n_features)
    self.bias = 0

    y_ = np.where(y > 0, 1, 0)

    # learn weights
    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)

        # perception update rule
        self.weights += self.lr * (y_[idx] - y_pred) * x_i
        self.bias += self.lr * (y_[idx] - y_pred)

    
  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_pred = self.activation_func(linear_output)
    return y_pred