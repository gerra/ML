from random import shuffle
import math
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

class Point:
    x = []
    label = 0

    def __init__(self, list, label=-1):
        self.x = list
        self.label = label

    def addDimension(self, f):
        xx = list(self.x)
        xx.append(f(self))
        return Point(xx, self.label)

    def convert(self, f):
        for i in range(0, len(x)):
            self.x[i] = f(self, i)

# metrics
def minkowskiDist(m, point1, point2):
    assert len(point1.x) == len(point2.x)
    s = 0
    for i in range(0, len(point1.x)):
        s += abs(point1.x[i] - point2.x[i]) ** m
    return s ** (1.0 / m)

def euclidDist(point1, point2):
    return minkowskiDist(2, point1, point2)

def manhattanDist(point1, point2):
    return minkowskiDist(1, point1, point2)


def keyFunction(point, dist):
    return lambda p: dist(point, p)

def distanceComparator(point, dist):
    def compare(point1, point2):
        return cmp(dist(point, point1), dist(point, point2))
    return compare

# cross-validations
def tkFoldCrossValidation(t, k, data, train):
    totalAccuracy = 0.0
    for it in range(0, t):
        shuffle(data)
        totalAccuracy += kFoldCrossValidation(k, data, train)
    return totalAccuracy / t

def kFoldCrossValidation(k, data, train):
    n = len(data)
    folds = list(chunks(data, k))
    totalAccuracy = 0.0
    for i in range(0, len(folds)):
        validateData = folds[i]
        chunked = folds[:i]
        if (i + 1) != n:
            chunked += folds[(i + 1):]
        trainData = unchunks(chunked)
        knn = train(trainData)
        totalAccuracy += accuracy(knn, validateData)
    return totalAccuracy / len(folds)

# accuracy
def accuracy(knn, data):
    correct = 0
    for point in data:
        if knn(point) == point.label:
            correct += 1
    return 1.0 * correct / len(data)

# utility
def chunks(data, k):
    m = int((len(data) + k - 1) / k)
    for i in range(0, len(data), m):
        yield data[i : i + m]

def unchunks(chunks):
    return [i for chunk in chunks for i in chunk]

# knn
def trainKnn(trainData, k, dist, weight):
    def knn(point):
        nn = sorted(trainData, key = keyFunction(point, dist))[:k]
        c0 = 0.0
        c1 = 0.0
        cnt0 = 0
        cnt1 = 0
        for p in nn:
            if p.label == 0:
                cnt0 += 1
                c0 += weight(dist(point, p))
            else:
                cnt1 += 1
                c1 += weight(dist(point, p))
        return 0 if c0 > c1 else 1
    return knn

def showData(data, k, dist, weight):
    offset = 0.05
    xMin = min([point.x[0] for point in data]) - offset
    xMax = max([point.x[0] for point in data]) + offset
    yMin = min([point.x[1] for point in data]) - offset
    yMax = max([point.x[1] for point in data]) + offset
    h = 0.05
    testX, testY = np.meshgrid(np.arange(xMin, xMax, h), np.arange(yMin, yMax, h))
    pairs = zip(testX.ravel(), testY.ravel())
    points = []
    for pair in pairs:
        points.append(Point([pair[0], pair[1]]))
    knn = trainKnn(data, k, dist, weight)
    classColormap  = ListedColormap(['#FF0000', '#00FF00'])
    testColormap   = ListedColormap(['#FFAAAA', '#AAFFAA'])
    pl.pcolormesh(testX,
                  testY,
                  np.asarray([knn(point) for point in points]).reshape(testX.shape),
                  cmap=testColormap)
    pl.scatter([point.x[0] for point in data],
               [point.x[1] for point in data],
               c = [point.label for point in data],
               cmap=classColormap)
    pl.show()

# IO
points = []
with open("chips") as f:
    for line in f:
        #points.append([float(part) for part in line.strip().split(',')])
        lst = [float(part) for part in line.strip().split(',')]
        points.append(Point(lst[0:len(lst)-1], lst[len(lst)-1]))

def normalize(data):
    ps = list(data)
    xMin = min([point.x[0] for point in data])
    xMax = max([point.x[0] for point in data])
    yMin = min([point.x[1] for point in data])
    yMax = max([point.x[1] for point in data])
    for p in ps:
        p.x[0] = (p.x[0] - xMin) / (xMax - xMin)
        p.x[1] = (p.x[1] - xMin) / (xMax - xMin)
    return ps

def transform(data):
    return [point.addDimension(lambda p: p.x[0] ** (1.0 / 3) + p.x[1] ** (1.0 / 3)) for point in data]

#print(normalize(points))
weight = lambda d: math.exp(-d)
#showData(normalize(points), 5, manhattanDist, weight)
#showData(points, 5, euclidDist)

metrics = [("euclid", euclidDist), ("manhattan", manhattanDist)]
for (mname, mm) in metrics:
    print(tkFoldCrossValidation(100, 11, transform(normalize(points)), lambda trainData: trainKnn(trainData, 5, mm, weight)))


"""
kInterval = range(4, 13)


for k in kInterval:
    print(k, end=',')

print()
for (mname, mm) in metrics:
    for foldQty in range(7, 13):
        print('-->{}'.format(foldQty))
        for k in kInterval:
            res = tkFoldCrossValidation(100, foldQty, normalize(points), lambda trainData: trainKnn(trainData, k, mm, weight))
            print(res, end = ',')
    print()
"""
