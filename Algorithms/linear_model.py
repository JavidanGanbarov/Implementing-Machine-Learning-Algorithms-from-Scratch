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



class LinearRegression:
    """
    Linear Regression: Used to model the linear relationship between independent variables (X)
    and a dependent variable (y).

    Model equation: y = Xw + b
    Where:
    w: Weights (coefficients for each independent variable),
    b: Bias (intercept).

    Step 1: Model Representation
    Add a bias term to the input matrix X (augmentation with a column of ones).

    Step 2: Optimization
    Use the Normal Equation to compute the weights and bias:
    w = (X^T * X)^(-1) * X^T * y
    Where X^T is the transpose of X, and X^(-1) is the inverse.

    Step 3: Predictions
    For new inputs, the predictions are made using:
    y_pred = X * w + b
    """

    def __init__(self):
        """
        Initialize the Linear Regression model parameters.
        """
        self.weights = None  # Model weights (excluding bias)
        self.bias = None  # Model bias
        self.coef_ = None  # Alias for weights
        self.rank_ = None  # Rank of the augmented X matrix
        self.singular_ = None  # Singular values of the augmented X matrix
        self.intercept_ = None  # Alias for bias
        self.n_features_ = None  # Number of features in the input data

    def fit(self, X, y):
        """
        Fit the Linear Regression model to the input data X and target values y.

        :param X: Input data (n_samples, n_features).
        :param y: Target values (n_samples,).
        """
        # Add a bias term to X by appending a column of ones
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute the transpose of X_bias
        X_transpose = X_bias.T

        # Calculate weights and bias using the Normal Equation
        self.weights = np.linalg.inv(X_transpose @ X_bias) @ X_transpose @ y

        # Separate bias and weights
        self.bias = self.weights[0]  # First value is the bias
        self.weights = self.weights[1:]  # Remaining values are the weights

        # Update model attributes
        self.coef_ = self.weights  # Alias for weights
        self.intercept_ = self.bias  # Alias for bias
        self.rank_ = np.linalg.matrix_rank(X_bias)  # Rank of the augmented X matrix
        self.singular_ = np.linalg.svd(X_bias, compute_uv=False)  # Singular values of X_bias
        self.n_features_ = X.shape[1]  # Number of features in the input data

    def predict(self, X):
        """
        Predict target values using the fitted Linear Regression model.

        :param X: Input data (n_samples, n_features).
        :return: Predicted values (n_samples,).
        """
        # Add a bias term to X by appending a column of ones
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute predictions
        return X_bias @ np.hstack([self.bias, self.weights])