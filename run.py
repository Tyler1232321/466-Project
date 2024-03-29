import numpy as np

import utilities as utils
import algorithms as algs
import random
import math
from sklearn.model_selection import KFold
from collections import defaultdict


def getaccuracy(ytest, predictions):
    correct = 0

    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    # count number of correct predictions
    #print( correct / len( ytest ) )
    #correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

def stratifiedCrossValidate(K, X, Y, Algorithm, parameters):
    from sklearn.model_selection import StratifiedKFold

    all_errors = np.zeros((len(parameters), K))

    skf = StratifiedKFold(n_splits=K, shuffle=True)

    count = 0
    for train_index, test_index in skf.split(X, Y):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        
        for i, params in enumerate(parameters):
            predictions = []
            learner = Algorithm(params)
            learner.learn(Xtrain, Ytrain)
            predictions = learner.predict(Xtest)
            print(geterror(Ytest, predictions))
            all_errors[i][count] = geterror(Ytest, predictions)
        count += 1
            

    print(all_errors)
    avg_errors = np.mean(all_errors, axis=1)
    min_error = 1000
    best_params = None
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])
        if avg_errors[i] < min_error:
            best_params = params
    return best_params


if __name__ == '__main__':

    classalgs = {
        #'Random': algs.Classifier,
        # 'Naive Bayes': algs.NaiveBayes,
        # 'Linear Regression': algs.LinearRegressionClass,
        #'Logistic Regression': algs.LogisticReg,
        #'Neural Network': algs.NeuralNet,
        #'BigNeuralNet': algs.BigNeuralNet,
        #'Kernel Logistic Regression': algs.KernelLogisticRegression,
        'SVM':algs.SVM
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
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
        ],
        'Naive Bayes': [
            { 'red_class_bias': 1.0 },
            { 'red_class_bias': 0.8 },
            { 'red_class_bias': 1.2 },
            { 'red_class_bias': 1.4 },
            { 'red_class_bias': 0.6 },
        ],
        'SVM': [
            {'kernel': 'linear'},
            {'kernel': 'rbf'},
            {'kernel': 'poly'},
            {'kernel': 'poly', 'degree': 5},
            {'kernel': 'poly', 'degree': 10},
            {'kernel': 'poly', 'degree': 15},
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = defaultdict(list)
    train_data = utils.load_data("true_train_data.csv")
    test_data = utils.load_data("true_test_data.csv")
    numruns = 2
    for learnername in classalgs:
        Xtrain = np.delete( train_data, 0, axis=1 ) # delete the last column, which are all Y values 
        Xtrain = Xtrain.astype(float)
        Ytrain = train_data[:, 0]
        Ytrain = Ytrain.astype(int)
        # cast the Y vector as a matrix
        Xtrain = np.reshape( Xtrain, [ len( Xtrain ), len( Xtrain[0] ) ] )
        Ytrain = np.reshape( Ytrain, [ len( Ytrain ), 1 ] )

        Xtest = np.delete( test_data, 0, axis=1 ) 
        
        Ytest = test_data[:, 0]
        # cast the Y vector as a matrix
        Xtest = np.reshape( Xtest, [ len( Xtest ), len( Xtest[0] ) ] )
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        Xtest = Xtest.astype(float)
        Ytest = Ytest.astype(int)

        # this section is for cross-validation
        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            best_parameters[learnername] = stratifiedCrossValidate(5, Xtrain, Ytrain, Learner, params)
        
        for r in range(numruns):
            # now we'll run the best set of parameters for each algorithm
            for learnername, Learner in classalgs.items():
                params = best_parameters[learnername]
                #print(params)
                #params = { 'epochs': 1000, 'nh1': 8 , 'nh2': 8}
                learner = Learner(params)
                learner.learn(Xtrain, Ytrain)
                predictions = learner.predict( Xtest )
                errors[learnername].append(geterror(Ytest, predictions))

    for learnername in classalgs:
        aveerror = np.mean(errors[learnername])
        stderror = np.std(errors[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))
        print ('Standard error for ' + learnername + ': ' + str( stderror / math.sqrt(numruns)) )
