PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100

########## ##########

# import pictures
import numpy as np
from mnist import MNIST
mndata = MNIST("./le4nn/")
X, Y = mndata.load_testing()
X = np.array(X)
# X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
Y = np.array(Y)

########## ##########

# X2 is 10000 * 784
X_size = X.shape[1]
N = X.shape[0]
# X2 = np.reshape(X, (X.shape[0], X_size))

np.random.seed(200)

########## ##########

# x is 10000 * 784 vector
def inputLayer(x):
    return x.reshape(-1, 1)

def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def sigmoid(x):
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

########## ##########

w1 = np.random.normal(0.0, 1.0 / X_size, (M_SIZE, X_size))
b1 = np.random.normal(0.0, 1.0 / X_size, (M_SIZE, 1))
w2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, M_SIZE))
b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE,1))

minibatch = np.random.choice(N, B_SIZE)

#print minibatch

averageOfEntropy = 0

for i in minibatch:
    inputX = X[i]
    x, y1, y2 = forward(inputX, w1, b1, w2, b2)
    y = [0] * Y[i] + [1] + [0] * (10 - Y[i] - 1)
    averageOfEntropy += crossEntropy(y, y2) / B_SIZE
    print '{0} : {1}'.format(recogRes(y2),Y[i])
    print crossEntropy(y, y2)

print averageOfEntropy
