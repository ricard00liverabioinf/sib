class NeuralNetwork:
    """
    Representa uma rede neuronal sequencial.
    """
    def __init__(self, layers: list, loss_function):
        self.layers = layers
        self.loss_function = loss_function

    def fit(self, X, y, epochs: int, lr: float):
        """Treina a rede neuronal utilizando o algoritmo de descida do gradiente."""
        for epoch in range(epochs):
            # 1. Forward propagation (passagem para a frente)
            output = X
            for layer in self.layers:
                output = layer.forward_propagation(output, training=True)
                
            # 2. Calcular o erro e o gradiente da perda
            # (Ex: MSE loss e respetiva derivada)
            loss = self.loss_function.loss(y, output)
            error = self.loss_function.derivative(y, output)
            
            # 3. Backward propagation (passagem para trás)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error)
                
            # 4. Atualização dos pesos (via otimizador ou gradiente descendente simples)
            for layer in self.layers:
                if hasattr(layer, 'weights') and layer.weights is not None:
                    layer.weights -= lr * layer.grad_weights
                    layer.biases -= lr * layer.grad_biases

    def predict(self, X):
        """Realiza previsões para novos dados."""
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training=False)
        return output