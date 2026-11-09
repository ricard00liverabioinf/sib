import numpy as np
from typing import List, Optional, Callable

class NeuralNetwork:
    """
    Classe NeuralNetwork que liga todas as camadas, funções de perda e otimizadores para treino e previsão[cite: 11, 12].
    """
    def __init__(self, 
                 epochs: int = 100, 
                 batch_size: int = 128, 
                 optimizer = None, 
                 learning_rate: float = 0.01, 
                 verbose: bool = False, 
                 loss = None, 
                 metric = None, 
                 **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.kwargs = kwargs
        
        self.layers: List = []
        self.history: dict = {}

    def add(self, layer):
        """
        Adiciona uma camada à rede neuronal, definindo o input_shape e inicializando com o otimizador[cite: 10].
        """
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].output_shape())
        
        if self.optimizer is not None:
            if isinstance(self.optimizer, type):
                opt_instance = self.optimizer(learning_rate=self.learning_rate, **self.kwargs)
            else:
                opt_instance = self.optimizer
            
            if hasattr(layer, 'initialize'):
                layer.initialize(opt_instance)
                
        self.layers.append(layer)

    def fit(self, dataset):
        """
        Treina a rede neuronal utilizando o dataset fornecido (ciclo por épocas e mini-batches)[cite: 12, 13].
        """
        X = dataset.X
        y = dataset.y
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.history = {'loss': [], 'metric': []}

        # Preparar a instância do otimizador base
        optimizer_instance = self.optimizer
        if isinstance(optimizer_instance, type):
            optimizer_instance = optimizer_instance(learning_rate=self.learning_rate, **self.kwargs)

        # Garantir que todas as camadas estão inicializadas
        for layer in self.layers:
            if hasattr(layer, 'initialize'):
                layer.initialize(optimizer_instance)

        for epoch in range(self.epochs):
            # 1. Baralhar os dados e dividir em mini-batches[cite: 11]
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            batches_X = np.array_split(X_shuffled, np.ceil(X.shape[0] / self.batch_size), axis=0)
            batches_y = np.array_split(y_shuffled, np.ceil(y.shape[0] / self.batch_size), axis=0)

            for batch_x, batch_y in zip(batches_X, batches_y):
                if batch_y.ndim == 1:
                    batch_y = batch_y.reshape(-1, 1)

                # 2. Forward propagation[cite: 11, 13]
                output = batch_x
                for layer in self.layers:
                    output = layer.forward_propagation(output, training=True)

                # 3. Calcular a derivada da perda[cite: 11, 13]
                error = self.loss.derivative(batch_y, output)

                # 4. Backward propagation (atualização de pesos por mini-batch)[cite: 11, 13]
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error)

            # 5. Calcular a perda e métrica global no final da época[cite: 11, 12]
            predictions = self.predict(dataset)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)

            epoch_loss = self.loss.loss(y, predictions)
            self.history['loss'].append(epoch_loss)

            epoch_metric = None
            if self.metric is not None:
                epoch_metric = self.metric(y, predictions)
                self.history['metric'].append(epoch_metric)

            # 6. Imprimir progresso se verbose for True[cite: 12]
            if self.verbose:
                metric_str = f" - Metric: {epoch_metric:.4f}" if epoch_metric is not None else ""
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}{metric_str}")

    def predict(self, dataset):
        """
        Faz previsões em novos dados utilizando forward propagation (com training=False)[cite: 13].
        """
        output = dataset.X
        for layer in self.layers:
            if hasattr(layer, 'forward_propagation'):
                try:
                    output = layer.forward_propagation(output, training=False)
                except TypeError:
                    output = layer.forward_propagation(output)
        return output

    def score(self, dataset):
        """
        Avalia a rede neuronal num dado dataset computando a métrica[cite: 13].
        """
        predictions = self.predict(dataset)
        y = dataset.y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        if self.metric is not None:
            return self.metric(y, predictions)
        return None