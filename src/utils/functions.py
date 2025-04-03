import numpy as np

def mse(X: np.ndarray, y: np.ndarray, model: np.ndarray, bias: np.array) -> float:
    return np.mean((X @ model + bias - y) ** 2)

def mae(X: np.ndarray, y: np.ndarray, model: np.ndarray, bias: np.array) -> float:
    return np.mean(np.abs(X @ model + bias - y))

def rmse(X: np.ndarray, y: np.ndarray, model: np.ndarray, bias: np.array):
    y_pred = X @ model + bias
    error = y_pred - y
    gradient = (2 / len(X)) * X.T @ error
    new_bias = np.mean(error)
    return gradient, new_bias

def rmae(X: np.ndarray, y: np.ndarray, model: np.ndarray, bias: np.array):
    y_pred = X @ model + bias
    error = np.sign(y_pred - y)
    gradient = (1 / len(y)) * X.T @ error
    new_bias = np.mean(error)
    return gradient, new_bias

def l1(model: np.ndarray) -> float:  # Lasso
    return np.sum(np.sign(model))

def l2(model: np.ndarray) -> float:  # Ridge
    return np.sum(2 * model)