import pandas as pd
import numpy as np
from sklearn.datasets import make_regression


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __repr__(self):
        params = ", ".join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'MyLineReg class: {params}'

    def add_bias(self, X):
        return pd.concat([pd.DataFrame({'bias': np.ones(X.shape[0])}), X], axis=1)

    def fit(self, X, y, verbose=False):
        # Add a column of ones on the left
        X = self.add_bias(X)

        # Initialize weights with ones
        self.weights = np.ones(X.shape[1])

        # Initial error
        predictions = self.predict(X, add_bias=False)
        errors = y - predictions
        mse = np.mean(errors ** 2)
        if verbose:
            print(f"start | loss: {mse:.2f}")

        # Gradient descent
        for i in range(self.n_iter):
            predictions = self.predict(X, add_bias=False)
            errors = y - predictions
            gradient = -2 * X.T.dot(errors) / len(y)
            self.weights -= self.learning_rate * gradient

            # Logging
            if verbose and (i + 1) % verbose == 0:
                mse = np.mean(errors ** 2)
                print(f"{i + 1} | loss: {mse:.2f}")

    def predict(self, X, add_bias=True):
        # Add a column of ones on the left only if necessary
        if add_bias:
            X = self.add_bias(X)
        return X.dot(self.weights)

    def get_coef(self):
        # Return weights without the first element (bias)
        return self.weights[1:]


# Example usage with make_regression
X, y = make_regression(n_samples=400, n_features=5, n_informative=5, noise=5, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)

# Train the model on the data
model = MyLineReg(n_iter=1000, learning_rate=0.01)
model.fit(X, y, verbose=100)

print("\nTrained model:")
print(model)

# New data for prediction
new_X, _ = make_regression(n_samples=10, n_features=5, n_informative=5, noise=5, random_state=42)
new_X = pd.DataFrame(new_X)

# Predicting values
predictions = model.predict(new_X)
print("\nNew data (X):")
print(new_X)
print("\nPredicted values:")
print(predictions)

# Get the coefficients (weights) of the model
coef = model.get_coef()
print("\nModel coefficients (weights):")
print(coef)

# Example usage for checking the sum of predictions
X_check, y_check = make_regression(n_samples=400, n_features=5, n_informative=5, noise=5, random_state=42)
X_check = pd.DataFrame(X_check)
y_check = pd.Series(y_check)

model_check = MyLineReg(n_iter=50, learning_rate=0.01)
model_check.fit(X_check, y_check, verbose=10)
predictions_check = model_check.predict(X_check)
print("\nSum of predictions for checking:")
print(predictions_check.sum())
