from random import shuffle
import math
import numpy as np
import matplotlib.pyplot as plt

class Point:
    x = []
    label = 0
    def __init__(self, list):
        dimension = len(list) - 1
        self.x = list[0:dimension]
        self.label = list[dimension]

    def addDimension(self, f):
        self.x.append(f(self))

    def convert(self, f):
        for i in range(0, len(x)):
            self.x[i] = f(self, i)

# metrics
def minkowskiDist(m, point1, point2):
    assert len(point1.x) == len(point2.x)
    s = 0
    for i in range(0, len(point1.x)):
        s += abs(point1.x[i] - point2.x[i]) ** m
    return s ** (1 / m)

def euclidDist(point1, point2):
    return minkowskiDist(2, point1, point2)

def manhattenDist(point1, point2):
    return minkowskiDist(1, point1, point2)


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
#    print('{} {}'.format(correct, len(data)))
    return 1.0 * correct / len(data)

# utility
def chunks(data, k):
    m = (len(data) + k - 1) / k
    for i in range(0, len(data), m):
        yield data[i : i + m]

def unchunks(chunks):
    return [i for chunk in chunks for i in chunk]

# knn
def trainKnn(trainData, k, dist):
    def knn(point):
        nn = sorted(trainData, cmp = distanceComparator(point, dist))[:k]
        c0 = 0
        c1 = 0
        for p in nn:
            if p.label == 0:
                c0 += 1
            else:
                c1 += 1
        return 0 if c0 > c1 else 1
    return knn

# IO
points = []
with open("chips") as f:
    for line in f:
        #points.append([float(part) for part in line.strip().split(',')])
        points.append(Point([float(part) for part in line.strip().split(',')]))

print('==={}==='.format(5))
print('--->{}<---'.format("manhatten"))
print(tkFoldCrossValidation(100, 10, points, lambda trainData: trainKnn(trainData, 5, manhattenDist)))

#metrics = [("euclid", euclidDist), ("manhatten", manhattenDist)]
#for k in range(1, 15):
#    print('==={}==='.format(k))
#    for (mname, mm) in metrics:
#        print('--->{}<---'.format(mname))
#        print(tkFoldCrossValidation(100, 10, points, lambda trainData: trainKnn(trainData, k, mm)))

