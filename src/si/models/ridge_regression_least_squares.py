import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression usando a solução analítica dos Mínimos Quadrados (Least Squares).
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale

        # Parâmetros estimados
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        X = dataset.X.copy()
        y = dataset.y

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        # Adicionar coluna de 1s para o termo de interseção (intercept)
        X_design = np.c_[np.ones(X.shape[0]), X]

        # Matriz de penalização L2 (não penaliza theta_zero)
        penalty_matrix = self.l2_penalty * np.eye(X_design.shape[1])
        penalty_matrix[0, 0] = 0

        # Fórmula analítica: (X^T * X + lambda * I)^(-1) * X^T * y
        thetas = np.linalg.inv(X_design.T.dot(X_design) + penalty_matrix).dot(X_design.T).dot(y)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.copy()

        if self.scale:
            X = (X - self.mean) / self.std

        X_design = np.c_[np.ones(X.shape[0]), X]
        thetas = np.r_[self.theta_zero, self.theta]

        return X_design.dot(thetas)

    def _score(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def score(self, dataset: Dataset) -> float:
        """Método público para garantir acesso direto ao cálculo do MSE."""
        return self._score(dataset)