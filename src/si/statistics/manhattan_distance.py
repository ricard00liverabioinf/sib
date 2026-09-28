import numpy as np


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula a distância de Manhattan (norma L1) entre um vetor x e uma matriz de vetores y.
    
    Formula: sum(|x_i - y_i|)
    """
    return np.sum(np.abs(x - y), axis=1)