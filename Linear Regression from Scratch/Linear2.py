import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = [] 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")

    def _compute_cost(self, y, y_predicted):
        n_samples = len(y)
        cost = (1 / (2 * n_samples)) * np.sum((y - y_predicted) ** 2)
        return cost
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)


