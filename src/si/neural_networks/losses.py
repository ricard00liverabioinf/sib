import numpy as np

class MSELoss:
    """
    Função de perda de Erro Quadrático Médio (Mean Squared Error).
    """
    def loss(self, y_true, y_pred):
        """Calcula o valor da perda (MSE)."""
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        """Calcula a derivada da perda em relação às previsões."""
        return -2 * (y_true - y_pred) / y_true.size