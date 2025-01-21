import numpy as np

class SimpleLinearRegression:

    """
    Simple Linear Regression: Used to model the linear relationship between an independent variable (x) and a dependent variable (y).

    Model equation: y = mx + b
    Where:
    m: Slope (represents the effect of x on y),
    b: Intercept (the point where the line crosses the y-axis).

    Step 1: Define the Loss Function
    The Mean Squared Error (MSE) is used to measure the difference between the predicted values and the actual values.
    MSE formula: J(m, b) = (1/n) * Σ (y_i - (mx_i + b))^2
    Goal: Minimize MSE to find the optimal values of m and b.

    Step 2: Optimization using Gradient Descent
    Gradient Descent is used to iteratively update m and b by calculating the gradients of the loss function.
    Gradients:
    For m: dJ/dm = -(2/n) * Σ x_i * (y_i - (mx_i + b))
    For b: dJ/db = -(2/n) * Σ (y_i - (mx_i + b))
    Update rules:
    m = m - α * (dJ/dm)
    b = b - α * (dJ/db)
    Where α is the learning rate.

    Step 3: Iterative Process
    Initialize m and b to 0 or random values.
    Compute gradients (dJ/dm and dJ/db).
    Update m and b using the learning rate and the computed gradients.
    Repeat steps 2 and 3 until the MSE value converges or a set number of iterations is completed.
    The final m and b values represent the best fit for the data.
    """


    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = None  # Slope (m)
        self.bias = None  # Intercept (b)
        self.cost_history = []

    def cost_function(self, X, y):
        """
        Calculate Cost Function (MSE).
        :param X: input data.
        :param y: true target values.
        :return: the mean squared error.
        """
        # Predicted value of y
        y_pred = np.dot(X, self.weight) + self.bias

        # Squared errors
        squared_errors = np.square(y - y_pred)

        # MSE (Mean Squared Error)
        mse = np.mean(squared_errors)

        return mse

    def predict(self, X):
        """
        Predict y.
        :param X: input data.
        :return: predicted values of y.
        """
        # Ensure that X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape to 2D if X is a 1D array

        # Predicted value of y
        y_pred = np.dot(X, self.weight) + self.bias

        return y_pred

    def fit(self, X, y):
        n = len(y)

        # Ensure X is 2D and y is 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 2:
            y = y.ravel()

        # Initialize parameters
        self.weight = np.zeros((X.shape[1], 1))
        self.bias = 0

        for _ in range(self.iterations):
            # Predicted values
            y_pred = self.predict(X)

            # Ensure y_pred is 1D to match y
            y_pred = y_pred.ravel()

            # Calculate gradients
            dW = -(2 / n) * np.dot(X.T, (y - y_pred)).reshape(-1, 1)  # Shape: (n_features, 1)
            dB = -(2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.weight -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            # Store the cost
            cost = self.cost_function(X, y)
            self.cost_history.append(cost)



