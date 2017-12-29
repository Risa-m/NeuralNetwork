PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100

ETA = 0.01

########## ##########

# import pictures
import numpy as np
from mnist import MNIST
mndata = MNIST("./le4nn/")
# testing or training
X, Y = mndata.load_testing()
X = np.array(X)
X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
Y = np.array(Y)

########## ##########

N = X.shape[0]
# X2 is 10000 * 784
X_size = X.shape[1] * X.shape[2] #784
XFlat = np.reshape(X, (N, X_size))

np.random.seed(200)


# first data
w1 = np.random.normal(0.0, 1.0 / X_size, (M_SIZE, X_size))
b1 = np.random.normal(0.0, 1.0 / X_size, (M_SIZE, 1))
w2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, M_SIZE))
b2 = np.random.normal(0.0, 1.0 / M_SIZE, (CLASS_SIZE, 1))

########## ##########


# x is 10000 * 784 vector
def inputLayer(x):
    return x.reshape(-1, 1)

def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def sigmoid(x):
    sigmoid_range = 34.538776394910684
    t = np.clip(x, -sigmoid_range, sigmoid_range)
    return 1.0 / (1.0 + np.exp(-t))

def softmax(a):
    alpha = max(a)
#    print np.sum(np.exp(a - alpha))
#    return np.exp(a * 100) / np.sum(np.exp(a * 100))
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha))

def recogRes(y):
    return np.argmax(y)

def forward(x, w1, b1, w2, b2):
    layer1Out = inputLayer(x)
    layer2In = fullyConnecterLayer(layer1Out, w1, b1)
#    print "#####"
#    print w1[1]
#    print b1
#    print layer2In
#    print "#####"
    layer2Out = sigmoid(layer2In)
    layer3In = fullyConnecterLayer(layer2Out, w2, b2)
    layer3Out = softmax(layer3In)
    return (layer1Out, layer2Out, layer3Out)

##########

def crossEntropy(AnsY, y):
    return (np.dot(AnsY, np.log(y))) * -1# + (np.dot(1 - np.array(AnsY), 1- np.log(y)))) * -1
    # return np.sum(np.dot(AnsY, np.log(y)))

##########

def backOfSoftAndCross(AnsY, y):
    return ((y.T - AnsY) / B_SIZE).reshape(-1, 1)

def backOfConnectX(w, diffY):
    return np.dot(w.T, diffY)

def backOfConnectW(x, diffY):
    return np.dot(diffY, x.T)

def backOfConnectB(diffY):
    return np.sum(diffY, axis=1).reshape(-1, 1)

def backOfConnect(w, x, diffY):
    deltaX = np.dot(w.T, diffY)
    deltaW = np.dot(diffY, x.T)
    deltaB = np.sum(diffY, axis=1).reshape(-1, 1)
    return (deltaX, deltaW, deltaB)

def backOfSig(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

########## ##########
"""
i = int(raw_input('>> '))
while i < 0 or i > 9999:
    i = int(raw_input('>> '))
"""
i = 0
########## ##########

count = 0
while count < 2 :
#while count < (N / B_SIZE * 10) :
    minibatch = np.random.choice(N, B_SIZE)

    averageOfError = 0

    deltaA = np.zeros(CLASS_SIZE).reshape(-1, 1)
    inputX2 = np.zeros(M_SIZE).reshape(-1, 1) # 50 * 1
    inputX1 = np.zeros(X_size).reshape(-1, 1)

    for i in minibatch:
        inputX = XFlat[i] / 255
        x, y1, y2 = forward(inputX, w1, b1, w2, b2)
        ansY = [0] * Y[i] + [1] + [0] * (10 - Y[i] - 1)
        averageOfError += crossEntropy(ansY, y2)
        deltaA = np.hstack((deltaA, backOfSoftAndCross(ansY, y2)))
        inputX1 = np.hstack((inputX1, x))
        inputX2 = np.hstack((inputX2, y1))
#        print '{0} : {1}'.format(recogRes(y2),Y[i])


    averageOfError /= B_SIZE

    deltaA = np.delete(deltaA, 0, axis=1)
    inputX2 = np.delete(inputX2, 0, axis=1)
    inputX1 = np.delete(inputX1, 0, axis=1)

    deltaX2, deltaW2, deltaB2 = backOfConnect(w2, inputX2, deltaA)
    deltaSig = backOfSig(deltaX2)
    _, deltaW1, deltaB1 = backOfConnect(w1, inputX1, deltaSig)

    w1 -= ETA * deltaW1
    w2 -= ETA * deltaW2
    b1 -= ETA * deltaB1
    b2 -= ETA * deltaB2

    # print averageOfError
    # 1e-15 ?
    print inputX2[0, 0]
#    print w1[0]
#    print deltaX2

    if (count % (N / B_SIZE)) == 0:
        print averageOfError
    count += 1


"""
import matplotlib.pyplot as plt
from pylab import cm
plt.imshow(X[i], cmap=cm.gray)
plt.show()
"""
