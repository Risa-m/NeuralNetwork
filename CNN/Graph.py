import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        precision = 0
        self.xAxis = []
        self.ce_height = []
        self.pr_height = []
        self.test_height = []

    def graphAppend(self, xAxis, crossEntropy, learningPrecision, testPrecision):
        self.xAxis.append(xAxis)
        self.ce_height.append(crossEntropy)
        self.pr_height.append(learningPrecision)
        self.test_height.append(testPrecision)

    def setData(self, xAxis, crossEntropy, learningPrecision, testPrecision):
        self.xAxis = xAxis
        self.ce_height = crossEntropy
        self.pr_height = learningPrecision
        self.test_height = testPrecision

    def getData(self):
        return (self.xAxis, self.ce_height, self.pr_height, self.test_height)

    def plot(self):
        print self.ce_height
        print self.pr_height
        print self.test_height
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        p1 = ax1.plot(self.xAxis, self.ce_height, color="lightskyblue", label="Cross Entropy")
        p2 = ax2.plot(self.xAxis, self.pr_height, color="crimson", label="learning data precision")
        p3 = ax2.plot(self.xAxis, self.test_height, color="limegreen", label="test data precision")
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('cross entropy')
        ax2.set_ylabel('precision')
        lns = p1+p2+p3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0, fontsize=8, ncol=3, mode="expand")
#        plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', borderaxespad=0)
        plt.subplots_adjust(top=0.9)
        plt.show()
