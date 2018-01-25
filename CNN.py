import Learning as le
import ReLU
import Pooling as po
import Convolution as co
from Graph import Graph
import numpy as np
from mnist import MNIST
import time
import os.path



PICT_HEIGHT = 28
PICT_WIDTH = 28
CLASS_SIZE = 10
M_SIZE = 50
B_SIZE = 100
ETA = 0.01
ALPHA = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8


R = 3
k1 = 8
d = 2

np.random.seed(200)


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

import cPickle
with open("le4nn/le4MNIST_X.dump","rb") as f:
    contestX = cPickle.load(f)
    contestX = contestX.reshape((contestX.shape[0], PICT_HEIGHT, PICT_WIDTH))
contest_size = 300

w1 = np.random.normal(0.0, 1.0 / (R * R * k1 * PICT_WIDTH * PICT_HEIGHT), (k1, R * R))
b1 = np.random.normal(1.0, 1.0 / (R * R * k1), (k1, 1))

w2 = np.random.normal(0.0, 1.0 / (M_SIZE), (M_SIZE, PICT_HEIGHT*PICT_WIDTH*k1/(d*d)))#
b2 = np.random.normal(1.0, 1.0 / (M_SIZE), (M_SIZE, 1))

w3 = np.random.normal(0.0, 1.0 / (CLASS_SIZE), (CLASS_SIZE, M_SIZE))
b3 = np.random.normal(1.0, 1.0 / (CLASS_SIZE), (CLASS_SIZE, 1))

mw = [0,0,0]
vw = [0,0,0]
mb = [0,0,0]
vb = [0,0,0]
adamT = 0

def fullyConnecterLayer(x, w, b):
    return np.dot(w, x) + b

def backOfConnect(x, w, deltaY):
    deltaX = np.dot(w.T, deltaY)
    deltaW = np.dot(deltaY, x.T)
    deltaB = (np.sum(deltaY, axis=1)/B_SIZE).reshape(-1,1)
    return (deltaX, deltaW, deltaB)

def softmax(a):
    alpha = np.max(a, axis=0)
    return np.exp(a - alpha) / np.sum(np.exp(a - alpha), axis=0)

def backOfSoftAndCross(ansY, y):
    return ((y - ansY) / B_SIZE)

def crossEntropy(AnsY, y):
    return np.sum(AnsY * np.log(y) * -1, axis=0)

def test():
    batchset = np.empty((p.shape[1] , p.shape[2], TEST_SIZE+contest_size))
    answer = np.array([q[i] for i in xrange(TEST_SIZE)])
    con_answer = [9,1,6,5,1,5,2,4,7,7,7,4,7,6,5,0,6,1,6,4,2,9,4,0,4,0,8,5,4,7,9,9,4,1,4,3,7,8,5,5,0,5,0,8,9,3,8,8,0,1,8,6,5,0,4,2,1,7,3,7,6,2,5,9,5,0,7,4,7,2,9,7,1,1,2,1,4,8,4,2,1,7,4,0,2,9,4,5,1,0,6,7,4,3,7,0,1,3,2,1,1,1,7,0,7,7,8,3,3,7,6,5,1,4,7,7,7,9,2,1,1,2,8,0,8,2,4,8,4,7,5,4,3,6,5,3,6,3,6,7,9,8,0,1,5,2,6,6,1,6,3,1,6,4,5,5,5,4,9,0,7,3,6,4,5,4,8,5,8,6,8,3,7,9,7,6,0,0,2,0,0,4,2,9,0,8,0,6,1,2,4,3,1,0,4,8,5,7,8,0,3,6,5,5,4,8,6,5,5,8,9,4,6,3,7,2,3,1,3,3,7,7,4,0,1,2,5,3,8,8,5,7,4,8,8,8,8,7,4,6,8,7,6,8,0,2,2,4,4,7,3,7,9,2,7,9,5,0,6,8,9,0,1,5,2,3,6,1,1,2,9,9,3,4,9,9,6,7,7,8,2,4,5,9,9,4,5,7,7,3,0,5,5,0,8,5,2,2,3,3]
    for i in xrange(TEST_SIZE+contest_size):
        if i < TEST_SIZE:
            batchset[:,:,i] = p[i]
        else:
            batchset[:,:,i] = contestX[i-TEST_SIZE]
    x1, y1 = c_test.convolution(batchset, w1, b1)
    z1 = relu.relu(y1)
    x2, choice1 = p_test.pooling(z1)
    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * (TEST_SIZE+contest_size))))
    x3 = relu.relu(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * (TEST_SIZE+contest_size))))
    res = softmax(y4)
    recog = np.argmax(res, axis=0)
    precision = len(np.where(answer - recog[0:TEST_SIZE] == 0)[0]) * 1.0 / TEST_SIZE
    contest_precision = len(np.where(con_answer - recog[TEST_SIZE:TEST_SIZE+contest_size] == 0)[0]) * 1.0 / contest_size
    print contest_precision
    return precision

def renewAdam(m, v, deltaW, t):
    m = BETA1 * m + (1 - BETA1) * deltaW
    v = BETA2 * v + (1 - BETA2) * deltaW * deltaW
    bm = m / (1 - np.power(BETA1, t))
    bv = v / (1 - np.power(BETA2, t))
    dW = ALPHA * bm / (np.sqrt(bv) + EPSILON)
    return (dW, m, v)


relu = ReLU.ReLU()
graph = Graph()

c1 = co.Convolution(PICT_HEIGHT, PICT_WIDTH, B_SIZE, R, k1)
p1 = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, B_SIZE, k1)
TEST_SIZE = q.shape[0]

c_test = co.Convolution(PICT_HEIGHT, PICT_WIDTH, TEST_SIZE+contest_size, R, k1)
p_test = po.Pooling(d, PICT_HEIGHT, PICT_WIDTH, TEST_SIZE+contest_size, k1)

"""
if(os.path.exists("CNNadam16-30.npz")):
    load_array = np.load("CNNadam16-30.npz")
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
    j = 0
    for i in choice:
        batchset[:,:,j] = X[i]
        ansY[:,j] = np.array([0] * answer[j] + [1] + [0] * (10 - answer[j] - 1))
        j += 1
    x1, y1 = c1.convolution(batchset, w1, b1)
    z1 = relu.relu(y1)
    x2, choice1 = p1.pooling(z1)
    y2 = fullyConnecterLayer(x2, w2, b2 * (np.array([1] * B_SIZE)))
    x3 = relu.sigmoid(y2)
    y4 = fullyConnecterLayer(x3, w3, b3 * (np.array([1] * B_SIZE)))
    res = softmax(y4)

    recog = np.argmax(res, axis=0)
    averageOfEntropy = np.sum(crossEntropy(ansY, res)) / B_SIZE
#    print len(np.where(answer - recog == 0)[0]) * 1.0 / B_SIZE
    precision += len(np.where(answer - recog == 0)[0]) * 1.0 / N
    if (count % (N / B_SIZE)) == 0:
        print count / (N / B_SIZE)
        print averageOfEntropy
        print precision
        testres = test()
        print testres
        graph.graphAppend(count / (N / B_SIZE), averageOfEntropy, precision, testres)

        precision = 0

    deltaSoft = backOfSoftAndCross(ansY, res)
#    print deltaSoft.shape #CLASS_SIZE, B_SIZE
    deltaX3, deltaW3, deltaB3 = backOfConnect(x3, w3, deltaSoft)
#    print deltaX3.shape #M_SIZE, B_SIZE
    deltaRelu2 = relu.backOfSig(x3, deltaX3)
    deltaX2, deltaW2, deltaB2 = backOfConnect(x2, w2, deltaRelu2)
#    print deltaX2.shape #k*width*height/d/d, B_SIZE
    deltaPooling = p1.backOfPooling(deltaX2, choice1)
    deltaRelu1 = relu.backOfReLU(z1, deltaPooling)
    deltaX1, deltaW1, deltaB1 = backOfConnect(x1, w1, deltaRelu1)
#    w1 -= (deltaW1 * ETA)
#    b1 -= (deltaB1 * ETA)
#    w2 -= (deltaW2 * ETA)
#    b2 -= (deltaB2 * ETA)
#    w3 -= (deltaW3 * ETA)
#    b3 -= (deltaB3 * ETA)

    adamT += 1
    dW1, mw[0], vw[0] = renewAdam(mw[0], vw[0], deltaW1, adamT)
    dW2, mw[1], vw[1] = renewAdam(mw[1], vw[1], deltaW2, adamT)
    dW3, mw[2], vw[2] = renewAdam(mw[2], vw[2], deltaW3, adamT)
    dB1, mb[0], vb[0] = renewAdam(mb[0], vb[0], deltaB1, adamT)
    dB2, mb[1], vb[1] = renewAdam(mb[1], vb[1], deltaB2, adamT)
    dB3, mb[2], vb[2] = renewAdam(mb[2], vb[2], deltaB3, adamT)
    w1 -= dW1
    w2 -= dW2
    w3 -= dW3
    b1 -= dB1
    b2 -= dB2
    b3 -= dB3


np.savez('CNNadam8-30s.npz', w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
#graph.plot()
"""
spl = np.split(deltaPooling, B_SIZE, axis=1)

import matplotlib.pyplot as plt
from pylab import cm
#t = spl[1].reshape(8,14,14)
t = spl[0].reshape(8, 28, 28)
#t = deltaPooling.reshape(8, 28, 28)
t = np.hstack((t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]))
#s = np.hstack((s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))
#u = np.vstack((t, s))
plt.imshow(t, cmap=cm.gray)
plt.show()
"""


"""
e1 = p1.batchtodimention(a1) ###check
f2 = np.empty((1, B_SIZE))
t = np.empty((k1*k2*PICT_WIDTH*PICT_HEIGHT/(d*d*d*d), B_SIZE))
inputx2 = np.empty((1, PICT_HEIGHT*PICT_HEIGHT*B_SIZE/(d*d)))
for i in xrange(e1.shape[0]):
    x2, y2 = c2.convolution(e1[i], w2, b2)
    z2 = relu.relu(y2)
    a2, choice2 = p2.pooling(z2)
    inputx2 = np.concatenate([inputx2, x2], axis=0)
    f2 = np.concatenate([f2, a2], axis=0)
f2 = np.delete(f2, 0, axis=0)
inputx2 = np.delete(inputx2, 0, axis=0)
res = fullyConnecterLayer(f2, w3, b3 * np.array([1]*B_SIZE))
res2 = softmax(res)
recog = np.argmax(res2, axis=0)
#print recog

deltaSoft = backOfSoftAndCross(answer, res2)
print deltaSoft.shape
deltaX3, deltaW3, deltaB3 = backOfConnect(f2, w3, deltaSoft)
print deltaX3.shape
print choice2.shape
backOfPooling2 = p2.backOfPooling(deltaX3, choice2)

"""
