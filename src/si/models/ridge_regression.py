import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5, scale: bool = True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Parâmetros estimados
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        X = dataset.X.copy()
        y = dataset.y.copy()

        # 1. Normalizar os dados se necessário
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0

        patience_count = 0
        last_cost = float('inf')

        for i in range(self.max_iter):
            # Previsão h_theta(x)
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - y

            # Atualizar os gradientes
            # theta_j := theta_j * (1 - alpha * lambda / m) - alpha * (1/m) * sum((y_pred - y) * x_j)
            grad_theta = (1 / m) * np.dot(X.T, error)
            self.theta = self.theta * (1 - self.alpha * (self.l2_penalty / m)) - self.alpha * grad_theta

            # theta_zero := theta_zero - alpha * (1/m) * sum(y_pred - y)
            grad_theta_zero = (1 / m) * np.sum(error)
            self.theta_zero = self.theta_zero - self.alpha * grad_theta_zero

            # Calcular função de custo J(theta)
            current_cost = self.cost(dataset, X_scaled=X)
            self.cost_history[i] = current_cost

            # Critério de paragem (patience)
            if last_cost - current_cost < 1e-4:
                patience_count += 1
                if patience_count >= self.patience:
                    break
            else:
                patience_count = 0

            last_cost = current_cost

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.copy()
        if self.scale and self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std

        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset, X_scaled: np.ndarray = None) -> float:
        if X_scaled is None:
            X = dataset.X.copy()
            if self.scale and self.mean is not None and self.std is not None:
                X = (X - self.mean) / self.std
        else:
            X = X_scaled

        m = X.shape[0]
        y_pred = np.dot(X, self.theta) + self.theta_zero
        cost_val = (1 / (2 * m)) * (np.sum((y_pred - dataset.y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2))
        return float(cost_val)