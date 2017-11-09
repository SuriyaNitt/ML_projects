#from __future__ import print_function
import numpy as np
import argparse
from copy import  copy

debug = 0;

def get_data(fileObject):
    lines = fileObject.readlines();
    numLines = len(lines);

    # Get the attribute names
    attributes = lines[0].split(',');
    attributes[-1] = attributes[-1].strip(' ');
    attributes[-1] = attributes[-1].strip('\r\n ');

    attributes = [attribute.strip(' ') for attribute in attributes];

    #print(attributes);

    data = [];
    # data parsing
    if numLines != 0:
        for i in range(1,numLines):
            lines[i] = lines[i].strip('\t');
            lines[i] = lines[i].strip('\n');
            lines[i] = lines[i].strip(' ');
            lineContent = lines[i].split(',');
            lineContent[-1] = lineContent[-1].strip('\r\n');
            data.append(lineContent);

    data = np.array(data);

    return (attributes, data[:, :-1], data[:, -1]);

def entropy(X, knowns=[], values=[]):
    sizeX = len(X);
    X = np.array(X);
    sizeKnowns = len(knowns);

    types = np.unique(X);
    numTypes = types.shape[0];

    mapTypes = {};
    for i, type in enumerate(types):
        mapTypes[type] = i;

    if sizeKnowns != 0:
        newX = [];
        for i in range(sizeX):
            valid = 1;
            for j in range(sizeKnowns):
                if knowns[j][i] != values[j]:
                    valid = 0;
            if valid:
                newX.append(X[i]);

        X = newX;

    X = np.array(X);
    sizeX = len(X);

    E = 0;
    countTypes = np.zeros((numTypes), dtype='uint32');

    if sizeX != 0:
        for i, x in enumerate(X):
            countTypes[mapTypes[x]] += 1;

        #if debug == 1:
        #    print countTypes;    

        for i, type in enumerate(types):
            numerator = countTypes[i];
            if numerator != 0:
                ratio = float(numerator) / float(sizeX);
                E -= ratio * np.log2(ratio);

    return E;

def joint_entropy(X, Y, knowns=[], values=[]):
    sizeX = len(X);
    sizeY = len(Y);

    assert(sizeX == sizeY);

    Z = [];

    for i in range(sizeX):
        z = str(X[i]) + str(Y[i]);
        Z.append(z);

    JE = entropy(Z, knowns, values);
    return JE;

def contitional_entropy(X, Y, knowns=[], values=[]):
    '''
    Implies H(X|Y)
    '''

    sizeX = len(X);
    sizeY = len(Y);

    assert(sizeX == sizeY);

    contitional_entropy = joint_entropy(X, Y, knowns, values) - entropy(Y, knowns, values);

    return contitional_entropy
    
def information_gain(X, Y, knowns=[], values=[]):
    sizeX = len(X);
    sizeY = len(Y);

    assert(sizeX == sizeY);

    E = entropy(X, knowns, values);
    CE = contitional_entropy(X, Y, knowns, values);
    IG = E - CE;
    #if debug == 1:
    #    print('E:{}, CE:{}'.format(E, CE));
    return IG;

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

def get_depth(tree, depth):
    if tree == None:
        return depth;
    elif tree.data == None:
        return depth;
    else:
        return max(get_depth(tree.left, depth+1), get_depth(tree.right, depth+1),);

def get_nodes(tree, nodes=[]):
    if tree == None:
        return nodes
    elif tree.data == None:
        return nodes
    else:
        nodes.append(tree.data[0]);
        leftNodes = get_nodes(tree.left, nodes)
        nodes += leftNodes;
        rightNodes = get_nodes(tree.right, nodes)
        nodes += rightNodes;
        return nodes

def get_label_distribution(yTrain, positives, negatives, knowns=[], values=[]):
    sizeKnowns = len(knowns);
    sizeY = len(yTrain);

    yTrain = np.array(yTrain);
    types = np.unique(yTrain);
    numTypes = types.shape[0];

    mapTypes = {'y': 1, 'n': 0};
    #for i, type in enumerate(types):
    #    mapTypes[type] = i;

    newYTrain = [];
    if sizeKnowns != 0:
        for i in range(sizeY):
            valid = 1;
            for j in range(sizeKnowns):
                if knowns[j][i] != values[j]:
                    valid = 0;
            
            if valid: 
                newYTrain.append(yTrain[i]);

        yTrain = newYTrain;

    yTrain = np.array(yTrain);

    count = np.zeros([numTypes], dtype='uint32');
    for i, y in enumerate(yTrain):
        count[mapTypes[y]] += 1;

    return (count[mapTypes[positives]], count[mapTypes[negatives]]);

def construct_BDT(tree, knownDists, path, attributesLeft, mapTrainAttributes, trainAttributes, positives, negatives, education):

    #print(attributesLeft);
    #print();
    knowns = []
    treeDepth = get_depth(BDT, 0);
    knowns = get_nodes(BDT, []);

    knowns = np.array(knowns);
    indexes = np.unique(knowns, return_index=True)[1];
    knowns = [knowns[index] for index in sorted(indexes)];
    if 'None' in knowns:
        knowns.remove('None');
    #knowns = list(set(knowns) - set([None]));

    if len(knowns) != 0 and len(path) != 0:
        knowns = knowns[:len(path)];

    if treeDepth > 2:
        knowns = knowns[:2];

    for known in knowns:
        if known in attributesLeft:
            attributesLeft.remove(known);

    #print(attributesLeft);
    #print();

    knownDists = [];
    knownValues = path;

    for i, known in enumerate(knowns):
        knownDists.append(mapTrainAttributes[known]);

    maxIG = 0;
    numPositives, numNegatives = get_label_distribution(mapTrainAttributes[trainAttributes[-1]], positives,
                                                        negatives, knownDists, knownValues);

    mapV = {}
    if education:
        mapV['y'] = 'A';
        mapV['n'] = 'notA';
    else:
        mapV['y'] = 'y';
        mapV['n'] = 'n';

    if len(path) < 3:
        if len(path) != 1:
            print('| {} = {}: [{}+/{}-]'.format(knowns[-1], mapV[path[-1]], numPositives, numNegatives));
        else:
            print('{} = {}: [{}+/{}-]'.format(knowns[-1], mapV[path[-1]], numPositives, numNegatives));

        #     print('| ', end='')
        # print(knowns[-1], end='');
        # print(' = {}: '.format(path[-1]), end='');
        # print('[{}+/{}-]'.format(numPositives, numNegatives));

    #if len(attributesLeft) != 0:
    if len(path) < 2 and len(attributesLeft) != 0:
        IGs = [information_gain(mapTrainAttributes[trainAttributes[-1]], mapTrainAttributes[attribute], knownDists,
                                knownValues) for attribute in attributesLeft];
        maxIG = 0;
        maxIGIndex = 0;
        if len(IGs) != 0:
            maxIG = np.max(IGs);
            maxIGIndex = np.argmax(IGs);
        if debug == 1:
            print(IGs);
        #print(attributesLeft);
        tree.data = [attributesLeft[maxIGIndex], numPositives, numNegatives];
    else:
        tree.data = ['None', numPositives, numNegatives];
    tree.right = Tree();
    tree.left = Tree();

    if numPositives != 0 and numNegatives != 0 and maxIG >= 0.1 and len(path) < 4:
        newPath = copy(path);
        newPath.append('y');

        construct_BDT(tree.right, knownDists, newPath, trainAttributes[:-1], mapTrainAttributes, trainAttributes, positives, negatives, education);

        newPath = copy(path);
        newPath.append('n');

        construct_BDT(tree.left, knownDists, newPath, trainAttributes[:-1], mapTrainAttributes, trainAttributes, positives, negatives, education);


def maxVote(myList):
    if myList[0] > myList[1]:
        return 1;
    else:
        return 0;

def evaluate(tree, map, i, positive, negative):
    treeData = tree.data[0];
    mapData = map[tree.data[0]][i];

    if map[tree.data[0]][i] == positive:
        if tree.right != None and tree.right.data != None:
            if tree.right.data[0] != 'None':
                return  evaluate(tree.right, map, i, positive, negative);
            else:
                return maxVote(tree.right.data[1:]);
        else:
            return maxVote(tree.data[1:]);

    else:
        if tree.left != None and tree.left.data != None:
            if tree.left.data[0] != 'None':
                return evaluate(tree.left, map, i, positive, negative);
            else:
                return maxVote(tree.left.data[1:]);
        else:
            return maxVote(tree.data[1:]);

def get_error(tree, map, names, data, label, positives, negatives, positive, negative):
    lenData = len(data);
    lenLabel = len(label);

    assert (lenData == lenLabel);

    mapLabel = {positives: 1, negatives: 0};
    correctPredictions = 0;

    for i, dataPoint in enumerate(data):
        prediction = evaluate(tree, map, i, positive, negative);
        if prediction == mapLabel[label[i]]:
            correctPredictions += 1

    errorRate = lenLabel - correctPredictions;
    errorRate = float(errorRate) / float(lenLabel);
    return  errorRate;

def view_tree(tree):
    if tree != None:
        print(tree.data);
        view_tree(tree.right);
        view_tree(tree.left);

if __name__ == '__main__':
    education = 0;
    #global politicians;
    
    # Parser object creation
    parser = argparse.ArgumentParser(description='Training and testing Decision Tree')
    parser.add_argument('train_file', type=str)
    parser.add_argument('test_file', type=str)

    # Parsing the input arguments
    args = parser.parse_args()

    # Opening the input files
    trainFileName = args.train_file
    trainFileObject = open(trainFileName, 'r')

    testFileName = args.test_file;
    testFileObject = open(testFileName, 'r');

    if 'education' in trainFileName:
        education = 1;

    # Read the content
    trainAttributes, xTrain, yTrain = get_data(trainFileObject);
    testAttributes, xTest, yTest = get_data(testFileObject);

    mapV = {'y': 'y', 'n':'n', 'yes':'y', 'no':'n', 'A':'y', 'notA':'n', 'democrat':'y', 'republican':'n'}
    mapV['before1950'] = 'y';
    mapV['after1950'] = 'n';
    mapV['morethan3min'] = 'y';
    mapV['lessthan3min'] = 'n';
    mapV['fast'] = 'y';
    mapV['slow'] = 'n';
    mapV['expensive'] = 'y';
    mapV['cheap'] = 'n';
    mapV['high'] = 'y';
    mapV['low'] = 'n';
    mapV['Two'] = 'y';
    mapV['MoreThanTwo'] = 'n';
    mapV['large'] = 'y';
    mapV['small'] = 'n';

    xTrainCols = xTrain.shape[1];
    for i in range(xTrainCols):
        for j in range(xTrain.shape[0]):        
            xTrain[j][i] = mapV[xTrain[j][i]];

    xTestCols = xTest.shape[1];
    for i in range(xTestCols):
        for j in range(xTest.shape[0]):        
            xTest[j][i] = mapV[xTest[j][i]];

    for j in range(yTrain.shape[0]):
        yTrain[j] = mapV[yTrain[j]];

    for j in range(yTest.shape[0]):
        yTest[j] = mapV[yTest[j]];


    if 'education' in trainFileName:
        education = 1;
    else:
        politicians = 1

    treeDepth = 0;
    attributesLeft = trainAttributes[:-1];
    
    mapTrainAttributes = {};
    mapTestAttributes = {};
    for i, attribute in enumerate(attributesLeft):
        mapTrainAttributes[attribute] = xTrain[:,i];
        mapTestAttributes[attribute] = xTest[:,i];

    mapTrainAttributes[trainAttributes[-1]] = yTrain;

    #print(trainAttributes);

    positives = 'y';
    negatives = 'n';
    positive = positives;
    negative = negatives;

    numPositives, numNegatives = get_label_distribution(mapTrainAttributes[trainAttributes[-1]], positives, negatives, [], []);
    print('[{}+/{}-]'.format(numPositives, numNegatives));

    basePath = [];
    BDT = Tree();

    knowns = []
    treeDepth = 0;
    attributesLeft = trainAttributes[:-1];
    knownDists = [];
    knownValues = basePath;
    IGs = [information_gain(mapTrainAttributes[trainAttributes[-1]], mapTrainAttributes[attribute], knownDists, knownValues) for attribute in attributesLeft];
    if debug == 1:
        print(IGs);
    maxIG = 0;
    maxIGIndex = 0;
    if len(IGs) != 0:
        maxIG = np.max(IGs);
        maxIGIndex = np.argmax(IGs);

    # print;
    # print(IGs);
    # print(attributesLeft);    
    # print;

    BDT.data = [attributesLeft[maxIGIndex], numPositives, numNegatives];
    BDT.right = Tree();
    BDT.left = Tree();
    path = copy(basePath);
    path.append('y');

    construct_BDT(BDT.right, [], path, trainAttributes[:-1], mapTrainAttributes, trainAttributes, positives, negatives, education);
    path = copy(basePath);
    path.append('n');

    construct_BDT(BDT.left, [], path, trainAttributes[:-1], mapTrainAttributes, trainAttributes, positives, negatives, education);

    errorTrain = get_error(BDT, mapTrainAttributes, trainAttributes, xTrain, yTrain, positives, negatives, positive, negative);
    errorTest = get_error(BDT, mapTestAttributes, testAttributes, xTest, yTest, positives, negatives, positive, negative);

    print('error(train): {}'.format(errorTrain));
    print('error(test): {}'.format(errorTest));
