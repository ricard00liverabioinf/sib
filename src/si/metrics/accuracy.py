import numpy as np

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia (precisão) entre os valores reais e previstos.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.round(np.asarray(y_pred)).flatten()
    return np.mean(y_true == y_pred)