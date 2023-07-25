import numpy as np

class NaiveBayes:
    
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)

    # calculate mean, var, and prior for each class
    self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
    self._var = np.zeros((n_classes, n_features), dtype=np.float64)
    self._priors = np.zeros(n_classes, dtype=np.float64)

    for c in self._classes:
      X_c = X[y == c]
      self._mean[c, :] = X_c.mean(axis=0)
      self._var[c, :] = X_c.var(axis=0)
      self._priors[c] = X_c.shape[0] / float(n_samples)

  def predict(self, X):
    pred = [self._predict(x) for x in X]
    return np.array(pred)
  
  def _predict(self, x):
    posteriors = []

    # calculate posterior probability for each class
    for idx, c in enumerate(self._classes):
      prior = np.log(self._priors[idx])
      posterior = np.sum(np.log(self._pdf(idx, x))) + prior
      posteriors.append(posterior)

    # return class with the highest posterior
    return self._classes[np.argmax(posteriors)]
      
  def _pdf(self, i, x):
    mean = self._mean[i]
    var = self._var[i]
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator
    
  
  