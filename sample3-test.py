PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
########## ##########

# import pictures
import os.path
import numpy as np
from mnist import MNIST
mndata = MNIST("./le4nn/")
X, Y = mndata.load_testing()
X = np.array(X)
Y = np.array(Y)

X_SIZE = X.shape[1]
N = X.shape[0]

np.random.seed(200)

########## ##########

w1 = np.random.normal(0.0, 1.0 / X_SIZE, (M_SIZE, X_SIZE))
b1 = np.random.normal(0.0, 1.0 / X_SIZE, (M_SIZE, 1))
w2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, M_SIZE))
b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE,1))

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

##########

def backOfSoftAndCross(ansY, y):
    return ((y.T - ansY) / B_SIZE).reshape(-1, 1)

def backOfConnect(x, w, deltaY):
    deltaX = np.dot(w.T, deltaY)
    deltaW = np.dot(deltaY, x.T)
    deltaB = np.sum(deltaY, axis=1).reshape(-1, 1)
    return (deltaX, deltaW, deltaB)

def backOfSig(x, deltaY):
    return (1 - x) * x * deltaY

########## ##########


filename = 'learningtest.npz'
if(os.path.exists(filename)):
    load_array = np.load(filename)
    w1 = load_array["w1"]
    b1 = load_array["b1"]
    w2 = load_array["w2"]
    b2 = load_array["b2"]
    load_array.close()


i = 0
precision = 0
while i < 100 :
    averageOfEntropy = 0
    correct = 0
    inputX = X[i] / 256.0
    x, y1, y2 = forward(inputX, w1, b1, w2, b2)
    ansY = [0] * Y[i] + [1] + [0] * (10 - Y[i] - 1)
    averageOfEntropy += crossEntropy(ansY, y2)
    correct = correct + 1.0 if(recogRes(y2) == Y[i]) else correct
    precision += correct / N
#    print averageOfEntropy
#    print '{0}'.format(correct*100)
#    print i
    print Y[i]
    import matplotlib.pyplot as plt
    from pylab import cm
    plt.imshow(X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))[i], cmap=cm.gray)
    plt.show()

    i += 1
print averageOfEntropy
print precision
