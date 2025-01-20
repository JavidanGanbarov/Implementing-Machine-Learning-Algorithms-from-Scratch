import numpy as np

"""
Here are some metrics for evaluating algorithms. 
Metrics are divided by 2 section: regression and classification metrics.
"""

"""Regression Metrics: MAE, MSE, RMSE, R-Squared, Adjusted R-Squared, Explained Variance Score"""


def mean_absolute_error(y_true, y_pred):

    """
    Calculate Mean Absolute Error (MAE).
    :param y_ture (numpy.ndarray): true target values.
    :param y_pred (numpy.ndarray): predicted target values.
    :return float: the mean absolute error.
    """

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Calculate mean absolute error
    mae = np.mean(abs_errors)

    return mae

def mean_squared_error(y_true, y_pred):

    """
    Calculate Mean Squared Error (MSE).
    :param y_ture (numpy.ndarray): true target values.
    :param y_pred (numpy.ndarray): predicted target values.
    :return float: the mean squared error.
    """

    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate squared errors
    squared_errors = np.square(y_true - y_pred)

    # Calculate mean sqared error
    mse = np.mean(squared_errors)

    return mse

def root_mean_squared_error(y_true, y_pred):

    """
    Calculate Root Mean Squared Error (RMSE).
    :param y_true (numpy.ndarray): true target values.
    :param y_pred (numpy.ndarray): predicted target values.
    :return float: the root mean squared error.
    """

    # Ensure inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate squared errors
    squared_errors = np.square(y_true - y_pred)

    # Calculate mean squared error
    mse = np.square(y_true - y_pred)

    # Calculate root mean squared error
    rmse = np.sqrt(mse)

    return rmse

def r2_score(y_true, y_pred):

    """
    Calculate R-squared (Coefficient of Determination)
    :param y_true (numpy.ndarray): true target values.
    :param y_pred (numpy.ndarray): predicted target values.
    :return float: r-squared score.
    """

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RSS and TSS
    rss = np.sum(np.square((y_true - y_pred)))
    tss = np.sum(np.square(y_true - np.mean(y_true)))

    # Calculate r-squared score
    r2 = 1 - (rss / tss)

    return r2

def adjr2_score(y_true, y_pred, n, k):

    """
    Calculate Adjusted R-Squared score.
    :param y_true (numpy.ndarray): true target values
    :param y_pred (numpy.ndarray): predicted target values
    :param n: number of data points in test set
    :param k: number of predictors
    :return float: adjusted r-squared score.
    """

    # Ensure inputs are arrays and integer
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = int(n)
    k = int(k)

    # Calculate R-Squared (Coefficient of Determination)
    rss = np.sum(np.square(y_true - y_pred))
    tss = np.sum(np.square(y_true - np.mean(y_true)))
    r2 = 1 - (rss / tss)

    # Calculate Adjusted R-Squared score
    r2_adj = 1 - ((1 - r2)*(n - 1)) / (n-k-1)

    return r2_adj


def explained_variance_score(y_true, y_pred):

    """
    Calculate Explained Variance Score (EVS).
    :param y_true (numpy.ndarrays): true target values
    :param y_pred (numpy.ndarrays): predicted target values
    :return float: explained variance score
    """

    # Ensure inputs are arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate residuals
    residuals = y_true - y_pred


    # Calculate variance of residuals
    var_residuals = np.var(residuals, ddof=1)

    # Calculate total variance
    total_var = np.var(y_true, ddof=1)

    # Calculate Explained Variance Score (EVS)
    evs = 1 - (var_residuals / total_var)

    return evs