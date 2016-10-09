from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as pl
import numpy as np
import random

class Point:
    x = []
    y = 0

    def __init__(self, list, y=-1):
        self.x = list
        self.y = y

def normalizePoints(points):
    if (len(points) == 0):
        return points
    dim = len(points[0].x)
    mins = [min(point.x[i] for point in points) for i in range(dim)]
    maxs = [max(point.x[i] for point in points) for i in range(dim)]
    newPoints = list(points)
    for point in newPoints:
        xs = point.x
        xs = [xs[i] if (maxs[i] - mins[i]) == 0 else (xs[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(dim)]
        point.x = xs
    return newPoints

def calcDirectly(xs, coefs):
    assert len(xs) == len(coefs)
    f = 0.0
    for i in range(len(xs)):
        f += coefs[i] * xs[i]
    return f

def calc(point, coefs):
    assert len(point.x) == len(coefs)
    return calcDirectly(point.x, coefs)

def getDeviations(points, coefs):
    n = len(points)
    deviations = [0.0 for i in range(n)]
    for i in range(n):
        deviations[i] = (calc(points[i], coefs) - points[i].y)
    return deviations

def calcAverageSquareDeviation1(deviations):
    res = 0.0
    for deviation in deviations:
        res += deviation ** 2
    n = len(deviations)
    res /= float(n)
    return res

def calcAverageSquareDeviation2(points, coefs):
    return calcAverageSquareDeviation1(getDeviations(points, coefs))

def useGradiendDescent(points):
    n = len(points)
    
    dim = len(points[0].x)
    w = [1.0 for i in range(dim)]
    w[len(w) - 1] = 0
    step = 0.0000002
    for counter in range(200000):
        difs = getDeviations(points, w)
        delta = calcAverageSquareDeviation1(difs)
        print(delta)
        #if (delta == 0):
        #    break
        for j in range(dim):
            sum = 0.0
            for i in range(n):
                sum += difs[i] * points[i].x[j]
            #oldW = w[j]
            w[j] -= step * (2.0 / n) * sum
    return w

def useGenetic(points, maxPopulationSize, survivalRate, mutantRate):
    def genCoef(i):
        return random.uniform(0, 30)
    def calcFitness(individual):
        coefs = individual
        return -calcAverageSquareDeviation2(points, coefs)
    def crossover(parent1, parent2):
        if (random.random() < 0.5):
            parent1, parent2 = parent2, parent1
        crossoverInd = random.randint(0, len(parent1))
        return (parent1[0:crossoverInd] + parent2[crossoverInd:len(parent2)])
    def getIndex(likelihoods):
        n = len(likelihoods)
        #val = random.random()
        #last = 0
        #for i in range(n):
        #    if last <= val and val <= likelihoods[i]:
        #        return i
        #    #else:
        #    #    last = likelihoods[i]
        return random.randint(0, n - 1)
    def createNewIndividual(population, likelihoods):
        populationSize = len(population)
        parent1Ind = 0
        parent2Ind = 0
        while parent1Ind == parent2Ind:
            #parent1Ind = random.randInt(0, populationSize - 1)
            #parent2Ind = random.randInt(0, populationSize - 1)
            parent1Ind = getIndex(likelihoods)
            parent2Ind = getIndex(likelihoods)
        #print("{} {} {}".format(parent1Ind, parent2Ind, populationSize))
        return crossover(population[parent1Ind], population[parent2Ind])
    def createNewPopulation(population, likelihoods, size):
        newPopulation = []
        for i in range(size):
            newPopulation.append(createNewIndividual(population, likelihoods))
        return newPopulation
    def mutate(population, count):
        for i in range(count):
            individual = population[i]
            m = random.randint(0, len(individual) - 1)
            individual[m] = genCoef(m)

    n = len(points)
    dim = len(points[0].x)
    population = [[genCoef(i) for i in range(dim)] for _ in range(maxPopulationSize)]
    for i in range(100):
        populationSize = len(population)
        fitnesses = [calcFitness(individual) for individual in population]

        # sort by indexes (i'm so lazy for non-two-array)
        indexes = range(populationSize)
        indexes = sorted(indexes, key = lambda ind: -fitnesses[ind])
        populationSize *= survivalRate
        population = [population[ind] for ind in indexes][:((int) (populationSize))]
        fitnesses  = [ fitnesses[ind] for ind in indexes][:((int) (populationSize))]

        # calculate likelihoods
        totalFitness = sum(fitnesses)
        averageFitness = totalFitness / populationSize
        likelihoods = map(lambda x: x / totalFitness, fitnesses)

        # create new population
        population = createNewPopulation(population, likelihoods, maxPopulationSize)
        newPopulationSize = len(population)

        # calculate average fitness of new population
        newAverageFitness = sum([calcFitness(k) for k in population]) / newPopulationSize

        # if new population is worse than previous, start mutation
        if newAverageFitness < averageFitness:
            np.random.shuffle(population)
            mc = max(1, mutantRate * newPopulationSize)
            mutate(population, (int) (mc))
        print(averageFitness)
    return population[0]

def drawPlane(ax, w, c):
    normal = [w[0], w[1], -1]
    d = -w[2]
    xx, yy = np.meshgrid(np.arange(0, 5500, 500), np.arange(0, 7, 1))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
    ax.plot_surface(xx, yy, z, alpha = 0.2, color = c)

# IO
if __name__ == '__main__':

    points = []
    with open("prices") as f:
        for line in f:
            lst = [float(part) for part in line.strip().split(',')]
            xs  = lst[0:len(lst) - 1]
            y   = lst[len(lst) - 1]
            xs  = xs + [1.0]
            y /= 100
            points.append(Point(xs, y))

    #w = [1.6309543300952616, 13.615201445927577, 7.420951549023872]

    geneticCoefs = []
    geneticError = 0
    for _ in range(10):
        w = useGenetic(points, 100, 0.5, 0.02)
        error = calcAverageSquareDeviation2(points, w)
        if len(geneticCoefs) == 0 or error < geneticError:
            geneticCoefs = w
            geneticError = error

    gradientCoefs = useGradiendDescent(points)
    gradientError = calcAverageSquareDeviation2(points, gradientCoefs)

    print("genetic coefficients: {}\ngenetic error: {}".format(geneticCoefs, geneticError))
    print("gradient coefficients: {}\ngradient error: {}".format(gradientCoefs, geneticError))

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = [point.x[0] for point in points]
    Y = [point.x[1] for point in points]
    Z = [point.y    for point in points]
    ax.scatter(X, Y, Z, c = 'r', marker = 'o')

    drawPlane(ax, geneticCoefs, 'blue')
    drawPlane(ax, gradientCoefs, 'green')
    pl.show()
