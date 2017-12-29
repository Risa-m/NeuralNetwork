PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50

########## ##########

# import pictures
import numpy as np
from mnist import MNIST
mndata = MNIST("./le4nn/")
X, Y = mndata.load_testing()
X = np.array(X)
X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
Y = np.array(Y)

########## ##########

X_size = X.shape[1] * X.shape[2]
X = np.reshape(X, (X.shape[0], X_size))

np.random.seed(200)

########## ##########


# x is 10000 * 784 vector
def inputLayer(x):
    return x.reshape(-1, 1)

def fullyConnecterLayer(w, b, x):
    return np.dot(w.T, x) + b

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def softmax(a):
    alpha = max(a)
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha))

def recogRes(y):
    return np.argmax(y)

########## ##########

i = int(raw_input('>> '))
while i < 0 or i > 9999:
    i = int(raw_input('>> '))

########## ##########

# i is max d
# j is max M
w1 = np.random.normal(0.0, 1.0 / X_size, (X_size, M_SIZE))
b1 = np.random.normal(0.0, 1.0 / X_size, (M_SIZE, 1))
w2 = np.random.normal(0.0, 1.0 / M_SIZE, (M_SIZE, CLASS_SIZE))
b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE,1))

x = inputLayer(X[i])
y1 = sigmoid(fullyConnecterLayer(w1, b1, x))
y2 = softmax(fullyConnecterLayer(w2, b2, y1))

print recogRes(y2)
