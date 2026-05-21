import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    X = np.asarray(X)
    X_standardize = np.array(X.shape)
    denominator = np.std(X, axis=axis, keepdims=True) + eps
    X_standardize = (X - np.mean(X, axis=axis, keepdims=True)) / denominator
    return X_standardize