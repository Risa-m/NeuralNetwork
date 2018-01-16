PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
########## ##########

# import pictures
import os.path
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST
mndata = MNIST("./le4nn/")
X, Y = mndata.load_training()
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

def backOfSig(t):
    return (1 - sigmoid(t)) * sigmoid(t)

########## ##########

"""
filename = 'test.npz'
if(os.path.exists(filename)):
    load_array = np.load(filename)
    w1 = load_array["w1"]
    b1 = load_array["b1"]
    w2 = load_array["w2"]
    b2 = load_array["b2"]
    load_array.close()
"""

count = 0
precision = 0
left = []
ce_height = []
pr_height = []
while count <= (N / B_SIZE * 100) :
    minibatch = np.random.choice(N, B_SIZE)
    averageOfEntropy = 0
    inputX1 = np.zeros(X_SIZE).reshape(-1, 1)
    inputX2 = np.zeros(M_SIZE).reshape(-1, 1)
    deltaA = np.zeros(CLASS_SIZE).reshape(-1, 1)
    correct = 0
    for i in minibatch:
        inputX = X[i] / 256.0
        x, y1, y2 = forward(inputX, w1, b1, w2, b2)
        ansY = [0] * Y[i] + [1] + [0] * (10 - Y[i] - 1)
        averageOfEntropy += crossEntropy(ansY, y2) / B_SIZE
        deltaA = np.hstack((deltaA, backOfSoftAndCross(ansY, y2)))
        inputX1 = np.hstack((inputX1, x))
        inputX2 = np.hstack((inputX2, y1))
        correct = correct + (1.0 / B_SIZE) if(recogRes(y2)==Y[i]) else correct
    deltaA = np.delete(deltaA, 0, axis=1)
    inputX2 = np.delete(inputX2, 0, axis=1)
    inputX1 = np.delete(inputX1, 0, axis=1)

    deltaX2, deltaW2, deltaB2 = backOfConnect(inputX2, w2, deltaA)
    deltaSig = (1 - inputX2) * inputX2 * deltaX2
    _, deltaW1, deltaB1 = backOfConnect(inputX1, w1, deltaSig)

    w1 -= (deltaW1 * ETA)
    b1 -= (deltaB1 * ETA)
    w2 -= (deltaW2 * ETA)
    b2 -= (deltaB2 * ETA)
    precision += correct / (N / B_SIZE)
#    print '{0}'.format(correct*100)
    if (count % (N / B_SIZE)) == 0:
        print averageOfEntropy
        print precision
        print count / (N / B_SIZE)
        left.append(count / (N / B_SIZE))
        ce_height.append(np.sum(averageOfEntropy))
        pr_height.append(precision)
        precision = 0
    count += 1

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(left, ce_height, color="lightskyblue")
ax2.plot(left, pr_height, color="crimson")
# print left
# print ce_height
ax1.set_xlabel('epoch')
ax1.set_ylabel('cross entropy')
ax2.set_ylabel('precision')
plt.show()
# np.savez('learning4.npz', w1=w1, b1=b1, w2=w2, b2=b2)

#[ 0.17225823]
#0.933583333333
