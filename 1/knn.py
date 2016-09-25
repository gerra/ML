import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Separate N data elements in two parts:
#	test data with N * testPercent elements
#	train_data with N * (1.0 - testPercent) elements
def splitToTrainAndTest(data, testPercent):
    trainData = []
    testData  = []
    for row in data:
        if random.random() < testPercent:
            testData.append(row)
        else:
            trainData.append(row)
    return trainData, testData

def euclidDist(point1, point2):
	return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def manhattenDist(point1, point2):
	return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def filter(points, id, third):
	return [point[id] for point in points if point[2] == third]

def distanceComparator(point, dist):
	def compare(point1, point2):
		d1 = dist(point, point1)
		d2 = dist(point, point2)
		if d1 < d2:
			return -1
		elif d1 > d2:
			return 1
		else: 
			return 0
	return compare

def leaveOneOut(trainData, dist):
	n = len(trainData)
	optimalK    = 1
	optimalFail = n
	for k in range(1, n):
		failCount = 0
		for i in range(0, n):
			point = trainData[i]
			points = trainData[0:i]
			if (i + 1 != n): 
				points += trainData[(i+1):n]
			pointClass = classify(point, points, k, dist)
			if pointClass != point[2]:
				failCount += 1
		if failCount < optimalFail:
			optimalFail = failCount
			optimalK    = k
	print(n)
	print(optimalFail)
	return optimalK


def getKNN(points, k, point, dist):
	return sorted(points, cmp = distanceComparator(point, dist))[:k]

def classify(pointToClassify, points, k, dist):
	knn = getKNN(points, k, pointToClassify, dist)
	c0  = 0
	c1  = 0
	for point in knn:
		if point[2] == 0:
			c0 += 1
		else:
			c1 += 1
	return 0 if c0 > c1 else 1


points = []
with open("chips") as f:
	for line in f:
		points.append([float(part) for part in line.strip().split(',')])

x0 = filter(points, 0, 0)
y0 = filter(points, 1, 0)

x1 = filter(points, 0, 1)
y1 = filter(points, 1, 1)

trainData, testData = splitToTrainAndTest(points, 0.2)
print(leaveOneOut(trainData, manhattenDist))

getKNN(points, 2, [-1,-1], euclidDist)

plt.plot(x0, y0, 'bo')
plt.plot(x1, y1, 'ro')
plt.show()