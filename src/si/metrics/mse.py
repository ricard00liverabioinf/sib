import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Mean Squared Error (MSE) entre os valores reais e previstos.
    """
    return float(np.mean((y_true - y_pred) ** 2))