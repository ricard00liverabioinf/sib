import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """
    Classe base abstrata para todas as funções de perda.
    Não necessita de __init__.
    """
    
    @abstractmethod
    def loss(self, y_true, y_pred):
        """Método abstrato que calcula a perda com base nos valores reais e previstos[cite: 5]."""
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """Método abstrato que calcula a derivada da perda em relação aos valores previstos[cite: 5]."""
        pass


class MeanSquaredError(LossFunction):
    """
    Função de perda de Erro Quadrático Médio (MSE) para tarefas de regressão[cite: 6].
    """
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size


class BinaryCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        # Evitar log(0) com np.clip
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.shape[0]


class CategoricalCrossEntropy(LossFunction):
    """
    Função de perda de Entropia Cruzada Categórica para problemas de classificação multi-classe[cite: 17].
    """
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred