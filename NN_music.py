import numpy as np
import scipy as sp
from scipy.special import expit
import random

import argparse

# Parser object creation
parser = argparse.ArgumentParser(description='Train and test NN for music dataset')
parser.add_argument('train_file', type=str)
parser.add_argument('train_labels', type=str)
parser.add_argument('test_file', type=str)

# Parsing the input arguments
args = parser.parse_args()

trainFileName         = args.train_file
trainLabelsFileName   = args.train_labels
testFileName          = args.test_file

debug = 0

def parse_file(fileName):
    fileObj         = open(fileName, 'r')
    lines           = fileObj.readlines()

    numLines        = len(lines)

    attributeNames  = []
    attributeValues = []

    for i, line in enumerate(lines):
        if i == 0:
            attributeNames = line.strip('\r').strip('\n').strip('\r').split(',')
        else:
            values         = line.strip('\r').strip('\n').strip('\r').split(',')
            values[2] = values[2].replace('yes', '1')
            values[3] = values[3].replace('yes', '1')
            values[2] = values[2].replace('no', '0')
            values[3] = values[3].replace('no', '0')
            values = [float(x) for x in values]
            values[0] = (values[0]-1900)/100
            values[1] = values[1]/7
            attributeValues.append(values)

    return (attributeNames, attributeValues)

def parse_labels(fileName):
    fileObj         = open(fileName, 'r')
    lines           = fileObj.readlines()

    numLines        = len(lines)

    labels          = list()

    for line in lines:
        line = line.strip('\r').strip('\n').strip('\r')
        line = line.replace('yes', '1')
        line = line.replace('no', '0')
        labels.append(float(line))

    return labels

def init_weights(numPrevUnits, numUnits):
    limits   = [-np.sqrt(6)/np.sqrt(numPrevUnits + numUnits), np.sqrt(6)/np.sqrt(numPrevUnits + numUnits)]
    weights  = np.random.uniform(limits[0], limits[1], [numPrevUnits, numUnits])
    bias     = np.zeros(numUnits)
    return (weights, bias)

def error(target, output):
    target = np.array(target)
    target = target.reshape((target.shape[0], 1))
    output = np.array(output)
    output = output.reshape((output.shape[0], 1))
    # for i, out in enumerate(output):
    #     if out > 0.7:
    #         output[i] = 1.0
    #     else:
    #         output[i] = 0.0
    err = 0.5 * (target - output) ** 2
    return err

class Layer():
    def __init__(self, neuronValues, numPrevUnits=0, numUnits=1, inputLayer=False, batchSize=1):
        self.numUnits     = numUnits
        self.neuronValues = neuronValues
        self.batchSize    = batchSize

        if not inputLayer:
            self.weights, self.bias  = init_weights(numPrevUnits, numUnits)

    def setValues(self, neuronValues):
        self.neuronValues = neuronValues

    def getWeights(self):
        return (self.weights, self.bias)

    def setWeights(self, weights):
        self.weights = weights

    def setBias(self, bias):
        self.bias = bias

class network():
    def __init__(self, inputs, numNeurons=5, batchSize=1):
        inputs            = np.array(inputs)
        self.inputs       = inputs
        self.numNeurons   = numNeurons
        self.batchSize    = batchSize

        inputSize         = inputs.shape[1]
        self.inputSize    = inputSize

        self.inputLayer        = Layer([], 0, numUnits=inputSize, inputLayer=True)
        self.hiddenLayer       = Layer([], numPrevUnits=inputSize, numUnits=numNeurons)
        self.outputLayer       = Layer([], numPrevUnits=numNeurons, numUnits=1)

    def fit(self, inputs, labels, lr=0.01, momentum=0.9):
        labels       = labels.reshape((labels.shape[0], 1))
        outputValues = self.predict(inputs)
        trainError   = error(labels, outputValues)
        velocity2    = np.zeros((self.numNeurons,1))
        velocity1    = np.zeros((self.inputSize, self.numNeurons))

        for i in range(labels.shape[0]):
            label        = labels[i]
            outputValue  = outputValues[i] 
            delW2        = -1 * lr * (label - outputValue)
            
            delW2        = delW2 * (outputValue - 1)
            delW2        = delW2 * outputValue

            temp         = delW2

            delB2        = delW2
            delW2        = delW2 * self.hiddenValues[i]

            delW1        = temp
            hiddenW, B   = self.outputLayer.getWeights()
            delB1        = delW1
            delW1        = delW1 * hiddenW
            dhdw         = (1 - self.hiddenValues[i]) * (self.hiddenValues[i])
            dhdw         = dhdw.reshape((dhdw.shape[0], 1))
            dhdw         = np.dot(dhdw, inputs[i].reshape(1, inputs.shape[1]))
            delB1        = delB1 * (1 - self.hiddenValues[i]) * (self.hiddenValues[i])
            delW1        = delW1 * dhdw

            W2, B2           = self.outputLayer.getWeights()
            W1, B1           = self.hiddenLayer.getWeights()

            velocity2        = ((momentum * velocity2) + delW2.reshape((delW2.shape[0], 1)))
            velocity1        = ((momentum * velocity1) + delW1.T)

            W2, B2           = W2 + velocity2, B2 + delB2.reshape((delB2.shape[0], 1))
            W1, B1           = W1 + velocity1, B1 + delB1

            self.outputLayer.setWeights(W2)
            self.hiddenLayer.setWeights(W1)

            self.outputLayer.setBias(B2)
            self.hiddenLayer.setBias(B1)

        outputValues = self.predict(inputs)
        trainError   = error(labels, outputValues)
        return trainError

    def predict(self, inputs):
        self.inputLayer.setValues(inputs)
        weights, bias = self.hiddenLayer.getWeights()
        hiddenValues  = np.dot(inputs, weights) + bias.reshape((1, bias.shape[0]))
        hiddenValues  = expit(hiddenValues)
        self.hiddenValues  = hiddenValues
        weights, bias = self.outputLayer.getWeights()
        outputValues  = np.dot(hiddenValues, weights) + bias.reshape((1, bias.shape[0]))
        outputValues  = expit(outputValues)

        # for i, out in enumerate(outputValues):
        #     if out > 0.2:
        #         outputValues[i] = 1.0
        #     else:
        #         outputValues[i] = 0.0
        return outputValues

def train(trainFileName, trainLabelsFileName):
    attributeNames, attributeValues = parse_file(trainFileName)
    labels                          = parse_labels(trainLabelsFileName)

    attributeValues                 = np.array(attributeValues)
    meanTrain                       = np.mean(attributeValues)
    varTrain                        = np.var(attributeValues)
    attributeValues                 -= meanTrain
    #attributeValues                 /= varTrain
    labels                          = np.array(labels).reshape(attributeValues.shape[0], 1)

    data                            = attributeValues
    data                            = np.append(data, labels, axis=1)
    np.random.shuffle(data)

    dataSize                        = data.shape[0]
    first90percent                  = int(np.floor(dataSize * 0.9))

    trainingData                    = data[:first90percent, :-1]
    trainingLabels                  = data[:first90percent, -1]
    validationData                  = data[first90percent:, :-1]
    validationLabels                = data[first90percent:, -1]

    validationError                 = 1
    trainError                      = 1

    NN                              = network(trainingData, 16)

    epoch                           = 1
    lr                              = 0.0012
    while trainError > 0.058 and epoch < 4000:
        lr              = lr - lr*1e-6
        trainError      = np.average(NN.fit(trainingData, trainingLabels, lr, momentum=0.9))
        prediction      = NN.predict(validationData)
        validationError = np.average(error(validationLabels, prediction))

        #print('Epoch:{}, TRE:{}, TE:{}'.format(epoch, trainError, validationError))
        print('{}'.format(trainError))
        epoch += 1

    return NN, meanTrain, varTrain

def test(testFileName, NN, meanTrain, varTrain):
    attributeNames, attributeValues = parse_file(testFileName)
    attributeValues -= meanTrain
    #attributeValues                 /= varTrain

    prediction = NN.predict(attributeValues)

    if debug == 1:
        testKeysFileName = './hw6/music_dev_keys.txt'
        labels           = parse_labels(testKeysFileName)
        testError        = error(labels, prediction)
        
        for i, value in enumerate(prediction):
            boolVal = 0.0
            if value > 0.5:
                boolVal = 1.0
            print('TrueValue:{}, prediction:{}, {}'.format(labels[i], value, boolVal))

        print np.average(testError)

    return prediction

if __name__ == '__main__':
    NN, meanTrain, varTrain = train(trainFileName, trainLabelsFileName)
    prediction = test(testFileName, NN, meanTrain, varTrain)

    print 'TRAINING COMPLETED! NOW PREDICTING.'

    for p in prediction:
        if p > 0.5: 
            print 'yes'
        else:
            print 'no'