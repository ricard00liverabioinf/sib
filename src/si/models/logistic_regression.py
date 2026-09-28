import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class LogisticRegression(Model):
    """
    Modelo de Regressão Logística com suporte para:
    - Regularização L1 (Lasso) e L2 (Ridge)
    - Gradient Descent com Momentum
    """

    def __init__(self, penalty: str = 'l2', alpha: float = 0.001, l2_penalty: float = 1.0, 
                 l1_penalty: float = 1.0, max_iter: int = 1000, momentum: float = 0.0, 
                 scale: bool = True):
        super().__init__()
        self.penalty = penalty  # 'l1', 'l2' ou None
        self.alpha = alpha  # Learning rate
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.momentum = momentum  # Fator de Momentum (0.0 desativa)
        self.scale = scale

        # Parâmetros estimados
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        X = dataset.X.copy()
        y = dataset.y
        m, n = X.shape

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Evita divisão por zero
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std

        self.theta = np.zeros(n)
        self.theta_zero = 0.0

        # Acumuladores para o Momentum
        v_theta = np.zeros(n)
        v_theta_zero = 0.0

        for i in range(self.max_iter):
            # Previsão das probabilidades com a função Sigmoide
            y_pred = self._sigmoid(X.dot(self.theta) + self.theta_zero)
            error = y_pred - y

            # Gradientes base
            grad_theta = (1 / m) * X.T.dot(error)
            grad_theta_zero = (1 / m) * np.sum(error)

            # Adição do termo de regularização
            if self.penalty == 'l2':
                grad_theta += (self.l2_penalty / m) * self.theta
            elif self.penalty == 'l1':
                grad_theta += (self.l1_penalty / m) * np.sign(self.theta)

            # Atualização dos parâmetros usando Momentum
            v_theta = self.momentum * v_theta + self.alpha * grad_theta
            v_theta_zero = self.momentum * v_theta_zero + self.alpha * grad_theta_zero

            self.theta -= v_theta
            self.theta_zero -= v_theta_zero

            # Custo (Binary Cross-Entropy) com penalização
            cost = (-1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            if self.penalty == 'l2':
                cost += (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
            elif self.penalty == 'l1':
                cost += (self.l1_penalty / (2 * m)) * np.sum(np.abs(self.theta))

            self.cost_history[i] = cost

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = dataset.X.copy()
        if self.scale:
            X = (X - self.mean) / self.std

        probs = self._sigmoid(X.dot(self.theta) + self.theta_zero)
        return (probs >= 0.5).astype(int)

    def _score(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def score(self, dataset: Dataset) -> float:
        return self._score(dataset)