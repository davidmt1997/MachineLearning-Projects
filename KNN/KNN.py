import sys
import random
import math
from scipy.io import arff
import numpy
import statistics
from sklearn.metrics import mean_squared_error


# load the arr file throw an exception if can't find the file
def loadDataFile(filename):
    try:
        data, meta = arff.loadarff(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit()
    return data


# load dataset and split it randomly in train and test set
def loadDataset(filename, split, trainingSet , testSet):
    try:
        data, meta = arff.loadarff(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit()

    for x in range(len(data)):
        if random.random() < split:
            trainingSet.append(data[x])
        else:
            testSet.append(data[x])

# finds teh euclidean distance for the 2 instances
def euclideanDistance(inst1, inst2):
    distance = 0
    for i in range(len(inst1)-1):
        if type(inst1[i]) is numpy.float64 and type(inst2[i]) is numpy.float64:
            distance += pow(inst1[i] - inst2[i], 2)
        else:
            if inst1[i] == inst2[i]:
                distance += 0
            else:
                distance += 1
    return math.sqrt(distance)


# returns an array with the k closest neighbors according to the euclidean distance
def getNeighbors(trainSet, testSetInst, k):
    distance = {}
    neighbors = []

    for j in range(len(trainSet)):
        dist = euclideanDistance(testSetInst, trainSet[j])
        distance[j] = dist
    sorted_x = sorted(distance.items(), key=lambda x: x[1])

    for x in range(len(k)):
        neighbors.append(sorted_x[x])
    return neighbors


# returns the mode of the given array
def getMode(array):
    return statistics.mode(array)


# returns the mean of the given array
def getMean(array):
    count = 0
    for i in range(len(array)):
        count += array[i]
    return count/len(array)


# returns the root mean squared error of the 2 arrays
def getRMSE(actual_val, predicted_val):
    return math.sqrt(mean_squared_error(actual_val, predicted_val))


# called when a training set and a testing set are provided
def mainWithArgs(trainingSet, testingSet, k):
    count = 0
    trainSet = loadDataFile(trainingSet)
    testSet = loadDataFile(testingSet)
    predictions = []
    actual =[]
    for i in range(len(testSet)):
        actual.append(testSet[i][len(testSet[i])-1])
        index = getNeighbors(trainSet, testSet[i], k)
        targetFeatures = []
        for j in range(len(index)):
            targetFeatures.append(trainSet[index[j][0]][len(trainSet[j]) - 1])
        print("test set: " + str(testSet[i][len(testSet[i])-1]))
        if type(targetFeatures[0]) is numpy.bytes_:
            print("predicted: " + str(getMode(targetFeatures)))
            predictions.append(getMode(targetFeatures))
        elif type(targetFeatures[0]) is numpy.float64:
            print("predicted: " + str(getMean(targetFeatures)))
            predictions.append(getMean(targetFeatures))

        if testSet[i][len(testSet[i])-1] == getMode(targetFeatures):
            print("CORRECT!")
            count +=1
        else:
            print("wrong")
    if type(trainSet[0][len(trainSet[i])-1]) is numpy.bytes_:
        print("Accuracy: " + str(count/len(testSet)*100) + "%")
    if type(trainSet[0][len(trainSet[i])-1]) is numpy.float64:
        print("Root mean squared error: " + str(getRMSE(predictions, actual)))


# called when only an arff file is provided
def mainNoArgs():
    trainSet = []
    testSet = []
    count = 0
    predictions = []
    actual = []
    filename = input('Enter your dataset: ')
    loadDataset(filename, 0.66, trainSet, testSet)

    kneighbors = input('Enter number of neighbors: ')
    while int(kneighbors) < 0:
        print("K has to be bigger than 0")
        kneighbors = input('Enter number of neighbors: ')
    for i in range(len(testSet)):
        actual.append(testSet[i][len(testSet[i]) - 1])
        index = getNeighbors(trainSet, testSet[i], kneighbors)
        targetFeatures = []
        for j in range(len(index)):
            targetFeatures.append(trainSet[index[j][0]][len(trainSet[j]) - 1])
        print("test set: " + str(testSet[i][len(testSet[i]) - 1]))
        if type(targetFeatures[0]) is numpy.bytes_:
            print("predicted: " + str(getMode(targetFeatures)))
            predictions.append(getMode(targetFeatures))
        elif type(targetFeatures[0]) is numpy.float64:
            print("predicted: " + str(getMean(targetFeatures)))
            predictions.append(getMean(targetFeatures))

        if testSet[i][len(testSet[i]) - 1] == getMode(targetFeatures):
            print("CORRECT!")
            count += 1
        else:
            print("wrong")
    if type(trainSet[0][len(trainSet[i])-1]) is numpy.bytes_:
        print("Accuracy: " + str(count/len(testSet)*100) + "%")
    if type(trainSet[0][len(trainSet[i])-1]) is numpy.float64:
        print("Root mean squared error: " + str(getRMSE(predictions, actual)))

if __name__ == "__main__":
    if len(sys.argv) == 4:
        trainingSet = sys.argv[1]
        testingSet = sys.argv[2]
        k = sys.argv[3]

        mainWithArgs(trainingSet, testingSet, k)
    else:
        mainNoArgs()
