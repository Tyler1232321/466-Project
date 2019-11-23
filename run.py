import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import algorithms as algs
import random
import math

from sklearn.model_selection import KFold


def getaccuracy(ytest, predictions):
    correct = 0

    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    print(correct)
    # count number of correct predictions
    #correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

def stratifiedCrossValidate(K, X, Y, Algorithm, parameters):
    from sklearn.model_selection import StratifiedKFold

    all_errors = np.zeros((len(parameters), K))

    skf = StratifiedKFold(n_splits=K)

    for train_index, test_index in skf.split(X, Y):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        count = 0
        for i, params in enumerate(parameters):
            learner = Algorithm(params)
            learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest)
            all_errors[i][count] = geterror(Ytest, predictions)
            count += 1

    avg_errors = np.mean(all_errors, axis=1)
    min_error = 1000
    best_params = None
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors < min_error:
            best_params = params
    return best_params


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=2000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=1000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=10,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset



    classalgs = {
        #'Random': algs.Classifier,
        #'Naive Bayes': algs.NaiveBayes,
        #'Linear Regression': algs.LinearRegressionClass,
        #'Logistic Regression': algs.LogisticReg,
        #'Neural Network': algs.NeuralNet,
        'BigNeuralNet': algs.BigNeuralNet,
        #'Kernel Logistic Regression': algs.KernelLogisticRegression,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Neural Network': [
            { 'epochs': 1000, 'nh': 4 },
            { 'epochs': 1000, 'nh': 8 },
            { 'epochs': 1000, 'nh': 12 },
            { 'epochs': 1000, 'nh': 16 },
        ],
        'BigNeuralNet': [
            { 'epochs': 1000, 'nh1': 4 , 'nh2': 4},
            { 'epochs': 1000, 'nh1': 8 , 'nh2': 8},
            { 'epochs': 1000, 'nh1': 16, 'nh2': 16},
            { 'epochs': 1000, 'nh1': 32, 'nh2': 32},
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        #print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        # this section is for cross-validation
        '''
        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)
        '''

        # now we'll run the best set of parameters for each algorithm
        for learnername, Learner in classalgs.items():
            #params = best_parameters[learnername]
            #print(params)
            params = { 'epochs': 1000, 'nh1': 8 , 'nh2': 8}
            learner = Learner(params)
            learner.learn(Xtrain, Ytrain)
            predictions = learner.predict( Xtest )
            errors[learnername][r] = geterror(Ytest, predictions)

    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))
        print ('Standard error for ' + learnername + ': ' + str( stderror / math.sqrt(numruns)) )
