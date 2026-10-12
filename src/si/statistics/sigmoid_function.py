import numpy as np

def sigmoid_function(X):
    """
    Calculates the sigmoid function for X.
    """
    return 1 / (1 + np.exp(-X))