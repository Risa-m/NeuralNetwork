import Learning as le
import ReLU
import Pooling as po
import Convolution as co
import numpy as np
from mnist import MNIST
import os.path
from Graph import Graph



PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
np.random.seed(200)


def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def backOfConnect(x, w, deltaY):
    deltaX = np.dot(w.T, deltaY)
    deltaW = np.dot(deltaY, x.T)
    deltaB = (np.sum(deltaY, axis=1)).reshape(-1,1)
    return (deltaX, deltaW, deltaB)

def softmax(a):
    alpha = np.max(a, axis=0)
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha), axis=0)

def backOfSoftAndCross(ansY, y):
    return ((y - ansY) / B_SIZE)

def crossEntropy(AnsY, y):
    return np.sum(AnsY * np.log(y) * -1, axis=0)


def test():
    batchset = np.empty((p.shape[1] , p.shape[2], TEST_SIZE))
    answer = np.array([q[i] for i in xrange(TEST_SIZE)])
    for i in xrange(TEST_SIZE):
        batchset[:,:,i] = p[i]
    x1, y1 = c_test.convolution(batchset, w1, b1)
    z1 = relu.relu(y1)
    x2 = np.empty((k1 * PICT_HEIGHT * PICT_WIDTH, TEST_SIZE))
    t = np.split(z1, TEST_SIZE, axis=1)
    for i in xrange(TEST_SIZE):
        x2[:,i] = t[i].ravel()
    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * TEST_SIZE)))
    x3 = relu.relu(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * TEST_SIZE)))
    res = softmax(y4)
    recog = np.argmax(res, axis=0)
    precision = len(np.where(answer - recog == 0)[0]) * 1.0 / TEST_SIZE
    return precision


mndata = MNIST("./le4nn/")
X, Y = mndata.load_training()
X = np.array(X)
N = X.shape[0]
X = X.reshape((X.shape[0],PICT_HEIGHT, PICT_WIDTH))
Y = np.array(Y)

p, q = mndata.load_testing()
p = np.array(p)
q = np.array(q)
p = p.reshape((p.shape[0], PICT_HEIGHT, PICT_WIDTH))


R = 3
k1 = 8
#k2 = 4
d = 2
w1 = np.random.normal(0.0, 1.0 / (R * R), (k1, R * R))
b1 = np.random.normal(0.0, 1.0 / (R * R), (k1, 1))

w2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, PICT_HEIGHT*PICT_WIDTH*k1))#
b2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, 1))

w3 = np.random.normal(0.0, 1.0 / (M_SIZE), (CLASS_SIZE, M_SIZE))
b3 = np.random.normal(0.0, 1.0 / (M_SIZE), (CLASS_SIZE, 1))

relu = ReLU.ReLU()
graph = Graph()
c1 = co.Convolution(PICT_HEIGHT, PICT_WIDTH, B_SIZE, R, k1)
p1 = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, B_SIZE, k1)
TEST_SIZE = q.shape[0]

c_test = co.Convolution(PICT_HEIGHT, PICT_WIDTH, TEST_SIZE, R, k1)
p_test = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, TEST_SIZE, k1)

"""
if(os.path.exists("convolution32-3.npz")):
    load_array = np.load("convolution32-3.npz")
    w1 = load_array["w1"]
    b1 = load_array["b1"]
    w2 = load_array["w2"]
    b2 = load_array["b2"]
    w3 = load_array["w3"]
    b3 = load_array["b3"]
    load_array.close()
"""


precision = 0
for count in xrange(N / B_SIZE * 30 + 1):
    choice = np.random.choice(X.shape[0], B_SIZE)
    batchset = np.empty((X.shape[1] , X.shape[2], B_SIZE))
    ansY = np.empty((CLASS_SIZE, B_SIZE))
    answer = np.array([Y[i] for i in choice])
    for i in xrange(B_SIZE):
        ansY[:,i] = np.array([0] * answer[i] + [1] + [0] * (10 - answer[i] - 1))
    j = 0
    for i in choice:
        batchset[:,:,j] = X[i]
        j += 1
    x1, y1 = c1.convolution(batchset, w1, b1)
    z1 = relu.relu(y1)

    x2 = np.empty((k1 * PICT_HEIGHT * PICT_WIDTH, B_SIZE))
    t = np.split(z1, B_SIZE, axis=1)
    for i in xrange(B_SIZE):
        x2[:,i] = t[i].ravel()

    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * B_SIZE)))
    x3 = relu.relu(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * B_SIZE)))
    res = softmax(y4)
#    print res
    averageOfEntropy = np.sum(crossEntropy(ansY, res)) / B_SIZE
#    print crossEntropy(ansY, res)
    recog = np.argmax(res, axis=0)
    precision += len(np.where(answer - recog == 0)[0]) * 1.0 / N
    if (count % (N / B_SIZE)) == 0:
#        recog = np.argmax(res, axis=0)
        print count / (N / B_SIZE)
        print averageOfEntropy
        print recog
        print answer
        print precision
        testres = test()
        print testres
        graph.graphAppend(count / (N / B_SIZE), averageOfEntropy, precision, testres)

        precision = 0

    deltaSoft = backOfSoftAndCross(ansY, res)
    deltaX3, deltaW3, deltaB3 = backOfConnect(x3, w3, deltaSoft)
    deltaRelu2 = relu.backOfReLU(x3, deltaX3)
    deltaX2, deltaW2, deltaB2 = backOfConnect(x2, w2, deltaRelu2)
    deltareshape = np.empty((k1, deltaX2.shape[0]*deltaX2.shape[1]/k1))
    t = np.split(deltaX2, B_SIZE, axis=1)
    for i in xrange(B_SIZE):
        deltareshape[:,i * PICT_WIDTH * PICT_HEIGHT : (i+1) * PICT_WIDTH * PICT_HEIGHT] = t[i].reshape(k1, -1)
#    deltX2 = deltaX2.reshape(k1, -1)
    deltaRelu1 = relu.backOfReLU(z1, deltareshape)
    deltaX1, deltaW1, deltaB1 = backOfConnect(x1, w1, deltaRelu1)

    w1 -= (deltaW1 * ETA)
    b1 -= (deltaB1 * ETA)
    w2 -= (deltaW2 * ETA)
    b2 -= (deltaB2 * ETA)
    w3 -= (deltaW3 * ETA)
    b3 -= (deltaB3 * ETA)


np.savez('convolution8-30.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
graph.plot()
"""
import matplotlib.pyplot as plt
from pylab import cm
t = a1.reshape(8,14,14)
s = choice1.reshape(8, 28, 28)
t = np.hstack((t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]))
s = np.hstack((s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
plt.imshow(s, cmap=cm.gray)
plt.show()
"""
