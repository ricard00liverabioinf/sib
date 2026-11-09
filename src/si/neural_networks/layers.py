from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(metaclass=ABCMeta):
    """
    Classe base abstrata para todas as camadas de uma rede neuronal.
    """
    
    @abstractmethod
    def forward_propagation(self, input, training):
        """Calcula a saída da camada dado o input."""
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        """Propaga o erro de saída para trás na rede."""
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        """Retorna a forma (shape) da saída da camada."""
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        """Retorna o número de parâmetros da camada."""
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        """Define a forma do input da camada."""
        self._input_shape = input_shape

    def input_shape(self):
        """Retorna a forma do input da camada."""
        return self._input_shape

    def layer_name(self):
        """Retorna o nome da classe da camada."""
        return self.__class__.__name__

class DenseLayer(Layer):
    """
    Camada densa (totalmente conectada) de uma rede neuronal.
    """
    def __init__(self, n_units: int, input_shape: tuple = None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.optimizer_weights = None
        self.optimizer_biases = None
        self.grad_weights = None
        self.grad_biases = None

    def initialize(self, optimizer=None):
        """Inicializa os pesos, os biases e cria instâncias independentes de otimizadores para cada um."""
        self.weights = np.random.uniform(-0.5, 0.5, (self._input_shape[0], self.n_units))
        self.biases = np.zeros((1, self.n_units))
        
        if optimizer is not None:
            opt_class = optimizer.__class__
            # Copiar os hiperparâmetros do otimizador (ex: learning_rate, momentum) sem o estado interno
            kwargs = {k: v for k, v in optimizer.__dict__.items() if k not in ['retained_gradient', 'm', 'v', 't']}
            self.optimizer_weights = opt_class(**kwargs)
            self.optimizer_biases = opt_class(**kwargs)
        return self

    def parameters(self) -> int:
        """Retorna o número de parâmetros da camada (pesos + biases)."""
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input, training: bool = False):
        """Calcula a propagação para a frente na camada densa."""
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        """Propaga o erro para trás na camada densa e atualiza os pesos e biases separadamente."""
        input_error = np.dot(output_error, self.weights.T)
        self.grad_weights = np.dot(self.input.T, output_error)
        self.grad_biases = np.sum(output_error, axis=0, keepdims=True)
        
        # Atualizar pesos e biases com otimizadores independentes
        if self.optimizer_weights is not None:
            self.weights = self.optimizer_weights.update(self.weights, self.grad_weights)
        if self.optimizer_biases is not None:
            self.biases = self.optimizer_biases.update(self.biases, self.grad_biases)
            
        return input_error

    def output_shape(self) -> tuple:
        """Retorna a forma da saída da camada."""
        return (self.n_units,)

class ReLU(Layer):
    """
    Camada de ativação ReLU (Rectified Linear Unit).
    """
    def __init__(self):
        self.input = None

    def forward_propagation(self, input, training: bool = False):
        """Aplica a função ReLU: max(0, input)."""
        self.input = input
        return np.maximum(0, input)

    def backward_propagation(self, output_error):
        """Propaga o erro através da derivada da ReLU."""
        return output_error * (self.input > 0)

    def output_shape(self) -> tuple:
        """Retorna a forma da saída (igual à forma do input)."""
        return self._input_shape

    def parameters(self) -> int:
        """Retorna o número de parâmetros (0 para a ReLU)."""
        return 0

class SigmoidActivation(Layer):
    """
    Camada de ativação Sigmoid.
    """
    def __init__(self):
        self.input = None

    def forward_propagation(self, input, training: bool = False):
        """Aplica a função Sigmoid."""
        self.input = input
        return 1 / (1 + np.exp(-np.clip(self.input, -500, 500))) # clip para estabilidade numérica

    def backward_propagation(self, output_error):
        """Propaga o erro através da derivada da Sigmoid."""
        sigmoid = 1 / (1 + np.exp(-np.clip(self.input, -500, 500)))
        return output_error * sigmoid * (1 - sigmoid)

    def output_shape(self) -> tuple:
        """Retorna a forma da saída (igual à forma do input)."""
        return self._input_shape

    def parameters(self) -> int:
        """Retorna o número de parâmetros (0 para ativações)."""
        return 0