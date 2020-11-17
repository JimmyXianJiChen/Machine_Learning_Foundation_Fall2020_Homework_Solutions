import numpy as np
from numpy import genfromtxt

def preprocess_data(x_0, scaling_factor=1):
    # preprocess the data
    raw_data = genfromtxt('./hw1_train.dat')
    X = raw_data[:, :10].copy()
    y = raw_data[:, 10].copy()
    # add x_0 into input
    X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1) * x_0 , X], axis=1) / scaling_factor
    return X, y


class perceptron():
    def __init__(self, X):
        self.W = np.zeros(X.shape[-1]) # initialize the weights of the perceptron
    def fit(self, X, y, rand_seed):
        # this function trains the perceptron

        def predict(X):
            # use the current perceptron to predict the class
            return np.sign(X @ self.W)
        
        def update_weights(x_n, y_n):
            # update the weights by: w_(t+1) = w_(t) + y_(n(t)) * x_(n(t))
            self.W = self.W + y_n * x_n
            return
        
        N = X.shape[0]
        np.random.seed(rand_seed)
        train_iter_cnt = 0 # number of updating
        predict_correct_cnt = 0 # number of consecutively correct predictions

        while predict_correct_cnt < 5 * N:
            # randomly pick a training instance
            n = np.random.randint(0, X.shape[0]) 
            x_t, y_t = X[n,:], y[n]
            if predict(x_t) != y_t:
                predict_correct_cnt = 0 # reset the counter
                update_weights(x_t, y_t)
                train_iter_cnt += 1
            else:
                predict_correct_cnt += 1
        return train_iter_cnt
