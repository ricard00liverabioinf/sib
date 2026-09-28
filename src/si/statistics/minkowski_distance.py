import numpy as np


def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float = 3.0) -> np.ndarray:
    """
    Calcula a distância de Minkowski entre um vetor x e uma matriz de vetores y.
    
    Formula: (sum(|x_i - y_i|^p))^(1/p)
    """
    return np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)