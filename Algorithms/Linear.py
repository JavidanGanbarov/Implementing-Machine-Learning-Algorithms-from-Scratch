import numpy as np
import scipy as sp

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def predict(self, X):
        X = self._add_bias(X)
        return np.dot(X, self.theta)

    def _compute_cost(self, X, y, theta):
        m = len(y)
        predictions = np.dot(X, theta)
        cost = np.sum(np.square(predictions - y)) / (2 * m)
        return cost

    def fit(self, X, y):

        return 0

    def gradient_descent(self, X, y):
        cost_history = []

        for i in range(self.iterations):




