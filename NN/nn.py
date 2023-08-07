from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
  return 1 / (1 + np.exp(-Z))

"""
w: weights, b: bias, i: input, h: hidden, o: output, l: label
w_i_h: weights from input layer to hidden layer
"""
images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

lr = 0.01
num_correct = 0
epochs = 3
for epoch in range(epochs):
  for img, l in zip(images, labels):
    # convert vectors to matrices
    img.shape += (1,)
    l.shape += (1,)

    # feed forward
    h = sigmoid(w_i_h @ img + b_i_h)
    o = sigmoid(w_h_o @ h + b_h_o)

    # cost using mse
    delta_o = o - l
    cost = 1 / len(o) * np.sum(delta_o ** 2, axis=0)
    num_correct += int(np.argmax(o) == np.argmax(l))

    # backpropagation output -> hidden
    w_h_o -= lr * delta_o @ np.transpose(h)
    b_h_o -= lr * delta_o
    # backpropagation hidden -> input
    delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
    w_i_h -= lr * delta_h @ np.transpose(img)
    b_i_h -= lr * delta_h

  # show accuracy for this epoch
  print(f"Accuracy: {round((num_correct / images.shape[0]) * 100, 2)}%")
  num_correct = 0

while True:
  index = np.random.randint(59999)
  img = images[index]
  plt.imshow(img.reshape(28, 28), cmap='Greys')

  img.shape += (1,)
  # forward prop input -> hidden
  h_pre = w_i_h @ img.reshape(784, 1) + b_i_h
  h = sigmoid(h_pre)
  # forward prop hidden -> output
  o_pre = w_h_o @ h + b_h_o
  o = sigmoid(o_pre)

  plt.title(f"This is number {o.argmax()}")
  plt.show()

