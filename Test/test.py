import numpy as np
from Metrics.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Algorithms.linear_model import LinearRegression

def create_synthetic_data(n_samples=100, n_features=2, noise=0.1):
    """
    Generates a synthetic dataset with a linear relationship.
    :param n_samples: Number of samples (rows)
    :param n_features: Number of features (columns)
    :param noise: Noise level (standard deviation of random noise)
    :return: (X, y, true_weights, true_bias): Features, target values, true weights, and true bias
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.rand(n_samples, n_features) * 10  # Random features in range [0, 10)
    true_weights = np.random.uniform(-5, 5, n_features)  # Random true weights
    true_bias = np.random.uniform(-5, 5)  # Random true bias

    # Linear relationship with added noise
    y = X @ true_weights + true_bias + np.random.normal(0, noise, size=n_samples)
    return X, y, true_weights, true_bias

# Generate synthetic data
X, y, true_weights, true_bias = create_synthetic_data(n_samples=200, n_features=3, noise=0.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("R^2 Score: {}".format(r2))
print("Mean Squared Error: {}".format(mse))

print("Coefficients: {}".format(model.coef_))

