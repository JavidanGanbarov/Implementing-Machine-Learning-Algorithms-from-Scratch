import numpy as np
from Algorithms.linear_model import SimpleLinearRegression
from Metrics.metrics import mean_squared_error, r2_score, adjr2_score
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature values between 0 and 10
y = 3 * X + np.random.randn(100, 1) * 2  # Linear relation with some noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Simple Linear Regression model
model = SimpleLinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: {}".format(mse))
print("R^2 Score: {}".format(r2))

