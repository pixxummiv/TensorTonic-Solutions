import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        y_pred = _sigmoid(np.dot(X, w) + b)

        # Compute gradients
        dw = np.dot(X.T, (y_pred - y)) / len(X)
        db = np.mean(y_pred - y)

        # Update parameters
        w = w - lr*dw
        b = b - lr*db

    return (w, b)
        