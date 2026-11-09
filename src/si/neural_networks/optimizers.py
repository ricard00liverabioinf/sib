import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Classe base abstrata para todos os otimizadores[cite: 8].
    """
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Método abstrato para atualizar os parâmetros (pesos ou bias) do modelo[cite: 8].
        """
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) com suporte a momentum[cite: 9, 10].
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        # Verificar se o retained_gradient está inicializado; caso contrário, inicializar com zeros[cite: 10]
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        # Computar e atualizar o gradiente retido (momentum) -> Vt = beta * Vt-1 + (1 - beta) * dL/dW
        # Nota: No contexto do slide, beta corresponde ao momentum.
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        # Atualizar os pesos -> Wt = Wt-1 - alpha * Vt
        w = w - self.learning_rate * self.retained_gradient
        return w


class Adam(Optimizer):
    """
    Otimizador Adam (Exercício 15), combinando ideias de RMSprop e SGD com momentum[cite: 18, 19, 20, 21].
    """
    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        # Verificar se m e v estão inicializados, se não, inicializar como matrizes de zeros[cite: 19]
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)

        # Atualizar o timestamp (epoch/step)
        self.t += 1

        # Computar e atualizar m e v[cite: 20]
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Computar a correção de bias (m_hat e v_hat)[cite: 21]
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Atualizar e retornar os pesos baseados nas médias móveis corrigidas[cite: 21]
        w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w