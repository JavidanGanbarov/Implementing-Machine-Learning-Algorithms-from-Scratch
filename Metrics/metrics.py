import numpy as np

"""
Here are some metrics for evaluating algorithms. 
Metrics are divided by 2 section: regression and classification metrics.
"""

"""Regression Metrics: MAE, MSE, RMSE, R-Squared, Adjusted R-Squared, Explained Variance Score"""


def mean_absolute_error(y_true, y_pred):

    """
    Calculate Mean Absolute Error (MAE).
    Parameters:
        y_true (numpy.ndarray): true target values.
        y_pred (numpy.ndarray): predicted target values.

        Returns:
            float: the mean absolute error
    """

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Calculate mean absolute error
    mae = np.mean(abs_errors)

    return mae



