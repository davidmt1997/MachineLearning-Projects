from scipy.io import arff
from collections import OrderedDict
import sys
import random
import numpy


# load the arr file throw an exception if can't find the file
def loadDataFile(filename):
    try:
        data, meta = arff.loadarff(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit()
    return data


# get the names of the attributes in the arff file
def getAttributeNames(filename):
    try:
        data, meta = arff.loadarff(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit()
    return meta.names()


# load the arff dataset and randomly split it into a training and a test set and return an array
# with the features of the dataset
# throws an exception if the features are not discrete
def loadDataset(filename, split, trainingSet , testSet):
    try:
        data, meta = arff.loadarff(filename)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit()

    for i in range(len(data)):
        for j in range(len(data[i])):
            if type(data[i][j]) is numpy.float64:
                raise Exception("Features must be discrete, use another dataset")

        if random.random() < split:
            trainingSet.append(data[i])
        else:
            testSet.append(data[i])
    return meta.names()


# counts how many of each feature there are in the dataset
def countValuesFromFeature(array, pos):
    values = OrderedDict()
    count = 1
    ret = []
    for i in range(len(array)):
        if array[i][pos] in values:
            values[array[i][pos]] += 1
        else:
            values[array[i][pos]] = count
    #print(values)
    ret = list(values.values())
    return ret


def getIndexFromFeature(array, pos, testFeature):
    values = OrderedDict()
    count = 1
    ret = []
    for i in range(len(array)):
        if array[i][pos] in values:
            values[array[i][pos]] += 1
        else:
            values[array[i][pos]] = count
    # print(values.items())
    ret = list(values.keys())
    #print(ret)
    for i in range(len(ret)):
        if ret[i] == testFeature:
            return i
    raise Exception("Index out of range exception")


#get a dictionary with the counts of all the target features
def getAllTargetFeatures(array):
    targets = OrderedDict()
    count = 1
    for i in range(len(array)):
        if array[i][len(array[i])-1] not in targets:
            targets[array[i][len(array[i])-1]] = count
        elif array[i][len(array[i])-1] in targets:
            targets[array[i][len(array[i])-1]] += 1

    return targets


#returns the number of target features
def getCountOfTargetFeatures(array):
    targets = OrderedDict()
    targets = getAllTargetFeatures(array)
    ret = list(targets.keys())
    return len(ret)

#returns the target feature at the given index
def getTargetFeatureAtIndex(array, index):
    targets = OrderedDict()
    targets = getAllTargetFeatures(array)
    ret = list(targets.keys())
    return ret[index]



# counts how many values of the particular target feature are for each feature in the dataset
def countTargetValuesFromFeature(array, pos, target):
    values = OrderedDict()
    count = 1
    ret = []
    for i in range(len(array)):
        if array[i][pos] in values and array[i][len(array[i])-1] == target:
            values[array[i][pos]] += 1
        elif array[i][pos] not in values and array[i][len(array[i])-1] == target:
            values[array[i][pos]] = count
        elif array[i][pos] not in values and array[i][len(array[i])-1] != target:
            values[array[i][pos]] = 0
    #print(values.items())
    ret = list(values.values())
    return ret


def getTargetFeatureProbability(array, targetFeature):
    count = 0
    total = len(array)

    for i in range(len(array)):
        if array[i][len(array[i])-1] == targetFeature:
            count += 1
    return count/total



#creates a probability table based on the counts of all the features and target features
def createProbabilityTable(countForFeatures, countForTargetFeatures, numTargetFeatures):
    table = []
    total = []
    for i in range(len(countForFeatures)):
        total.append(countForFeatures[i])
    table.append(total)

    for i in range(numTargetFeatures):
        fractions = []
        for j in range(len(countForFeatures)):
            #print(str(countForTargetFeatures[i][j]) + "/" + str(total[j]))
            fractions.append(countForTargetFeatures[i][j]/total[j])
        table.append(fractions)

    #print(table)
    return table


def mainWithArgs(trainingSet, testingSet):
    trainSet = loadDataFile(trainingSet)
    testSet = loadDataFile(testingSet)
    totalCountForTargetFeature = []
    totalCountForFeature = []
    attributeNames = getAttributeNames(trainingSet)
    probabilityTable = []

    for i in range(len(attributeNames)-1):
        totalCountForFeature.append(countValuesFromFeature(trainSet, i))
        for j in range(getCountOfTargetFeatures(trainSet)):
            totalCountForTargetFeature.append(
                countTargetValuesFromFeature(trainSet, i, getTargetFeatureAtIndex(trainSet, j)))

    print(totalCountForFeature)
    print(totalCountForTargetFeature)



    for i in range(len(attributeNames)-1):
        probabilityTable.append(createProbabilityTable(totalCountForFeature[i], totalCountForTargetFeature, getCountOfTargetFeatures(trainSet)))
        for j in range(getCountOfTargetFeatures(trainSet)):
            #print("removing: " + str(totalCountForTargetFeature[0]))
            totalCountForTargetFeature.remove(totalCountForTargetFeature[0])

    # re inserting the values in totalCountForTargetFeature
    for j in range(getCountOfTargetFeatures(trainSet)):
        totalCountForTargetFeature.append(
            countTargetValuesFromFeature(trainSet, i, getTargetFeatureAtIndex(trainSet, j)))

    #print(probabilityTable[0])
    print(probabilityTable)

    #print(getIndexFromFeature(trainSet, 1, b'partlyCloudy'))
    #print(getTargetFeatureProbability(totalCountForFeature, totalCountForTargetFeature, 0, 0))

    for i in range(len(testSet)):
        print(testSet[i])
    count = 0
    for i in range(len(testSet)):
        actual = testSet[i][len(testSet[i]) - 1]
        #print(actual)
        probablities = []
        for z in range(getCountOfTargetFeatures(trainSet)):
            print("target feature: " + str(getTargetFeatureAtIndex(trainSet, z)) + " target feature prob: " + str(getTargetFeatureProbability(trainSet, getTargetFeatureAtIndex(trainSet, z))))
            prob = getTargetFeatureProbability(trainSet, getTargetFeatureAtIndex(trainSet, z))
            for j in range(len(testSet[i])-1):
                print("probability of: " + str(testSet[i][j]) + " " + str(probabilityTable[j][z+1][getIndexFromFeature(trainSet, j, testSet[i][j])]))
                prob *= probabilityTable[j][z+1][getIndexFromFeature(trainSet, j, testSet[i][j])]
            probablities.append(prob)
            print("probabilities: " + str(probablities))
        mostProbable = probablities[0]
        index = 0
        for i in range(1, len(probablities)):
            if(probablities[i] > mostProbable):
                mostProbable = probablities[i]
                index = i

        predicted = getTargetFeatureAtIndex(trainSet, index)
        #print("index: " + str(index))
        print("actual: " + str(actual))
        print("predicted: " + str(predicted))
        if actual == predicted:
            print("CORRECT!")
            count += 1
    print("accuracy: " + str(count/len(testSet)*100) + "%")

def mainWithNoArgs():
    trainSet = []
    testSet = []
    totalCountForFeature = []
    totalCountForTargetFeature = []
    probabilityTable = []

    filename = input('Enter your dataset: ')
    attributeNames = loadDataset(filename, 0.66, trainSet, testSet)

    for i in range(len(attributeNames)-1):
        totalCountForFeature.append(countValuesFromFeature(trainSet, i))
        for j in range(getCountOfTargetFeatures(trainSet)):
            totalCountForTargetFeature.append(
                countTargetValuesFromFeature(trainSet, i, getTargetFeatureAtIndex(trainSet, j)))

    print(totalCountForFeature)
    print(totalCountForTargetFeature)



    for i in range(len(attributeNames)-1):
        probabilityTable.append(createProbabilityTable(totalCountForFeature[i], totalCountForTargetFeature, getCountOfTargetFeatures(trainSet)))
        for j in range(getCountOfTargetFeatures(trainSet)):
            #print("removing: " + str(totalCountForTargetFeature[0]))
            totalCountForTargetFeature.remove(totalCountForTargetFeature[0])

    # re inserting the values in totalCountForTargetFeature
    for j in range(getCountOfTargetFeatures(trainSet)):
        totalCountForTargetFeature.append(
            countTargetValuesFromFeature(trainSet, i, getTargetFeatureAtIndex(trainSet, j)))

    #print(probabilityTable[0])
    print(probabilityTable)

    #print(getIndexFromFeature(trainSet, 1, b'partlyCloudy'))
    #print(getTargetFeatureProbability(totalCountForFeature, totalCountForTargetFeature, 0, 0))

    for i in range(len(testSet)):
        print(testSet[i])
    count = 0
    for i in range(len(testSet)):
        actual = testSet[i][len(testSet[i]) - 1]
        #print(actual)
        probablities = []
        for z in range(getCountOfTargetFeatures(trainSet)):
            print("target feature: " + str(getTargetFeatureAtIndex(trainSet, z)) + " target feature prob: " + str(getTargetFeatureProbability(trainSet, getTargetFeatureAtIndex(trainSet, z))))
            prob = getTargetFeatureProbability(trainSet, getTargetFeatureAtIndex(trainSet, z))
            for j in range(len(testSet[i])-1):
                print("probability of: " + str(testSet[i][j]) + " " + str(probabilityTable[j][z+1][getIndexFromFeature(trainSet, j, testSet[i][j])]))
                prob *= probabilityTable[j][z+1][getIndexFromFeature(trainSet, j, testSet[i][j])]
            probablities.append(prob)
            print("probabilities: " + str(probablities))
        mostProbable = probablities[0]
        index = 0
        for i in range(1, len(probablities)):
            if(probablities[i] > mostProbable):
                mostProbable = probablities[i]
                index = i

        predicted = getTargetFeatureAtIndex(trainSet, index)
        #print("index: " + str(index))
        print("actual: " + str(actual))
        print("predicted: " + str(predicted))
        if actual == predicted:
            print("CORRECT!")
            count += 1
    print("accuracy: " + str(count/len(testSet)*100) + "%")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        trainingSet = sys.argv[1]
        testingSet = sys.argv[2]
        mainWithArgs(trainingSet, testingSet)
    else:
        mainWithNoArgs()