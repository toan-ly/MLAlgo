import numpy as np
from collections import Counter

class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
  
  def is_leaf_node(self):
    return self.value is not None
  
class DecisionTree:
  def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features  
    self.root = None 
    
  def fit(self, X, y):
    self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
    self.root = self._grow_tree(X, y)

  def _grow_tree(self, X, y, depth=0):
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))
    
    # check the stopping criteria
    if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
      leaf_value = self._most_common_label(y)
      return Node(value=leaf_value)
    
    feat_idx = np.random.choice(n_feats, self.n_features, replace=False)
    
    # find the best split
    best_thres, best_feature = self._best_split(X, y, feat_idx)
    
    # create child nodes
    left_idx, right_idx = self._split(X[:, best_feature], best_thres)
    left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
    right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
    return Node(best_feature, best_thres, left, right)
  
  def _most_common_label(self, y):
    return Counter(y).most_common()[0][0]
  
  def _best_split(self, X, y, feat_idx):
    best_gain = -1
    split_idx, split_threshold = None, None
    
    for i in feat_idx:
      X_column = X[:, i]
      thresholds = np.unique(X_column)

      for thr in thresholds:
        # calculate the information gain
        gain = self._information_gain(y, X_column, thr)

        if gain > best_gain:
          best_gain = gain
          split_idx = i
          split_threshold = thr
    
    return split_threshold, split_idx

  def _information_gain(self, y, X_column, threshold):
    # parent entropy
    parent_entropy = self._entropy(y)
    
    # create children
    left_idx, right_idx = self._split(X_column, threshold)
    
    if len(left_idx) == 0 or len(right_idx) == 0:
      return 0
    
    # calculate the weighted avg entropy of children
    n = len(y)
    n_left, n_right = len(left_idx), len(right_idx)
    e_left, e_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])
    child_entropy = n_left / n * e_left + n_right / n * e_right
    
    # calculate the IG 
    information_gain = parent_entropy - child_entropy 
    return information_gain
    
  def _split(self, X_column, threshold):
    left = np.argwhere(X_column <= threshold).flatten()
    right = np.argwhere(X_column > threshold).flatten()
    return left, right
    
  def _entropy(self, y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log(p) for p in ps if p > 0])
  
  def predict(self, X):
    return np.array([self._traverse_tree(x, self.root) for x in X])

  def _traverse_tree(self, x, node):
    if node.is_leaf_node():
      return node.value

    if x[node.feature] <= node.threshold:
      return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right)