import numpy as np

import utilities as utils
import random

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # red_class_bias is an additional bias we place in favour of the red fighter,
        # where 1 means no bias, 2 means we will only choose blue if it is twice as likely as red, 
        # 0.5 means that we will only choose red is it is twice as likely as blue, and so on
        self.params = utils.update_dictionary_items({'red_class_bias': 1}, parameters)

    # helper function to seperate data by class
    def seperate_data(self, xtrain, ytrain):
        n, m = xtrain.shape
        seperated_data = {}
        for i in range(n):
            if ytrain[i][0] not in seperated_data:
                seperated_data[ytrain[i][0]] = []

            seperated_data[ytrain[i][0]].append(xtrain[i])
        return seperated_data

    def learn(self, Xtrain, ytrain):
        self.red_bias = self.params['red_class_bias']

        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')

        n,m = Xtrain.shape

        # get the probabilities of each class occuring
        bincount = np.bincount( ytrain.astype(int)[:, 0]) 
        self.probabilities = []
        for b in bincount:
            self.probabilities.append(b / n)

        # seperate the data by class
        self.seperated_data = self.seperate_data(Xtrain, ytrain)

        # get the class stats for each feature
        self.class_stats = []
        for i in range(self.numclasses):
            self.class_stats.append({})

            for feature in range(m):
                self.class_stats[i][feature] = {}

                self.class_stats[i][feature]['feat_mean'] = \
                        np.mean([x[feature] for x in self.seperated_data[i]] )
                self.class_stats[i][feature]['standard_dev'] = \
                        np.std([x[feature] for x in self.seperated_data[i]] )

    def predict(self, Xtest):

        n,m = Xtest.shape
        # initialize predictions list
        predictions = []
        # for each point
        for i in range(n):
            maxInd = -1
            maxProb = -1000000
            # calculate the probability of each class occuring given the data point
            for k in range( self.numclasses):
                prob = self.probabilities[k]
                for j in range(m):
                    val1 = Xtest[i][j]
                    val2 = self.class_stats[k][j]["feat_mean"]
                    val3 = self.class_stats[k][j]['standard_dev']
                    # print(val1, val2, val3, type(val1), type(val2), type(val3))
                    prob *= utils.gaussian_pdf( \
                        Xtest[i][j], \
                        self.class_stats[k][j]["feat_mean"], \
                        self.class_stats[k][j]['standard_dev'] )
                # if we are looking at red's probability, add red's bias
                if k == 1:
                    prob *= self.red_bias
                # if this class is most probable so far, set it as best
                if prob > maxProb:
                    maxProb = prob
                    maxInd = k
            # choose the best one
            predictions.append(maxInd)
        #return predictions
        return np.reshape(predictions, [n, 1])


class BigNeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh1': 4,
            'nh2': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.001,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None

    # cost function for testing
    def cost(self, X, y):
        preds = self.predict(X)
        tot_err = 0
        for i in range(len(preds)):
            tot_err += abs(preds[i] - y[i])
        return tot_err

    def learn(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.wh1 = np.random.rand( m, self.params['nh1'] );
        self.wh2 = np.random.rand( self.params['nh1'], self.params['nh2'] )
        self.wo = np.random.rand( self.params['nh2'], 1 );

        step_size = self.params['stepsize']

        # Batch gradient descent with backpropagation
        for i in range(self.params['epochs']):

            hidden_layer1 = self.transfer( np.dot( Xtrain, self.wh1 ) )
            hidden_layer2 = self.transfer( np.dot( hidden_layer1, self.wh2 ) )
            output = self.transfer( np.dot( hidden_layer2, self.wo ) )

            output_error = output - ytrain

            output_gradient = np.dot(hidden_layer2.T, output_error)

            temp = hidden_layer2 * ( 1 - hidden_layer2 )
            hidden_error_2 = np.multiply( self.wo.dot(output_error.T).T , temp )
            hidden_gradient_2 = hidden_layer1.T.dot(hidden_error_2)

            temp = hidden_layer1 * ( 1 - hidden_layer1 )
            hidden_error_1 = np.multiply( self.wh2.dot( hidden_error_2.T ).T, temp )
            hidden_gradient_1 = Xtrain.T.dot( hidden_error_1 )

            
            for n2 in range(len(self.wo)):
                self.wo[n2][0] -= output_gradient[n2] * step_size
                for n1 in range( len( self.wh2 ) ):
                    self.wh2[n1][n2] -= hidden_gradient_2[n1][n2] * step_size

            for feature in range( len( self.wh1 ) ):
                for n1 in range( len( self.wh2 ) ):
                    self.wh1[feature][n1] -= hidden_gradient_1[feature][n1] * step_size

    def predict(self,Xtest):
        predictions = []
        preds = []
       
        hidden_layer1 = self.transfer( np.dot( Xtest, self.wh1 ) )

        hidden_layer2 = self.transfer( np.dot( hidden_layer1, self.wh2 ) )

        output = self.transfer( np.dot( hidden_layer2, self.wo ) )

        for val in output:
            if val > 0.5:
                predictions.append(1.)
            else:
                predictions.append(0.)
                
        return predictions

# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.001,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None

    # for testing purposes
    def cost(self, X, y):
        preds = self.predict(X)
        tot_err = 0
        for i in range(len(preds)):
            tot_err += abs(preds[i] - y[i])
        return tot_err

    
    def learn(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.wi = np.random.rand( m, self.params['nh'] );
        self.wo = np.random.rand( self.params['nh'], 1 );

        step_size = self.params['stepsize']

        # Batch gradient descent with back propagation
        for i in range(self.params['epochs']):

            hidden_layer = self.transfer( np.dot( Xtrain, self.wi ) )
            output = self.transfer( np.dot( hidden_layer, self.wo ) )

            error = output - ytrain

            output_gradient = np.dot(hidden_layer.T, error)

            temp = hidden_layer * (1 - hidden_layer)
            hidden_error = np.multiply( self.wo.dot(error.T).T , temp )

            hidden_gradient = Xtrain.T.dot(hidden_error)
            
            for neuron in range(len(self.wo)):
                self.wo[neuron][0] -= output_gradient[neuron] * step_size
                for feature in range(len(self.wi)):
                    self.wi[feature][neuron] -= hidden_gradient[feature][neuron] * step_size

    def predict(self,Xtest):
        predictions = []
        preds = []
       
        hidden_layer = self.transfer( np.dot( Xtest, self.wi ) )

        output = self.transfer( np.dot( hidden_layer, self.wo ) )


        for val in output:
            if val > 0.5:
                predictions.append(1.)
            else:
                predictions.append(0.)
        return predictions
