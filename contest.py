PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50

########## ##########

# import pictures

import os.path
import numpy as np
import cPickle
with open("le4nn/le4MNIST_X.dump","rb") as f:
    X = cPickle.load(f)
X = X.reshape((X.shape[0], 784))

#X, Y = mndata.load_testing()
#X = np.array(X)
#Y = np.array(Y)

X_SIZE = X.shape[0]
np.random.seed(200)

########## ##########

# x is 10000 * 784 vector
def inputLayer(x):
    return x.reshape(-1, 1)

def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def sigmoid(x):
    sigmoid_range = 34.538776394910684
    t = np.clip(x, -sigmoid_range, sigmoid_range)
    return 1 / (1 + np.exp(-x))

def softmax(a):
    alpha = max(a)
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha))

def recogRes(y):
    return np.argmax(y)

def forward(x, w1, b1, w2, b2):
    layer1Out = inputLayer(x)
    layer2In = fullyConnecterLayer(layer1Out, w1, b1)
    layer2Out = sigmoid(layer2In)
    layer3In = fullyConnecterLayer(layer2Out, w2, b2)
    layer3Out = softmax(layer3In)
    return (layer1Out, layer2Out, layer3Out)

##########

def crossEntropy(AnsY, y):
    return np.dot(AnsY, np.log(y)) * -1


#w1 = np.random.normal(0.0, 1.0 / X_SIZE, (M_SIZE, X_SIZE))
#b1 = np.random.normal(0.0, 1.0 / X_SIZE, (M_SIZE, 1))
#w2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, M_SIZE))
#b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE,1))

filename = 'learningtest.npz'
if(os.path.exists(filename)):
    load_array = np.load(filename)
    w1 = load_array["w1"]
    b1 = load_array["b1"]
    w2 = load_array["w2"]
    b2 = load_array["b2"]
    load_array.close()

##########
i = 0
while i < 100:
    inputX = X[i] / 256.0
    x, y1, y2 = forward(inputX, w1, b1, w2, b2)
    resY = recogRes(y2)

    print resY
#    print Y[i]

    import matplotlib.pyplot as plt
    from pylab import cm
    plt.imshow(X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))[i], cmap=cm.gray)
    plt.show()
    i += 1
