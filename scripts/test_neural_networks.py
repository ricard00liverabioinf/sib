import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from si.neural_networks.layers import DenseLayer, ReLU
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.losses import MSELoss

# 1. Gerar dados sintéticos para regressão (y = 2*x1 + 3*x2 + 1)
np.random.seed(42)
X = np.random.randn(100, 2)
y = 2 * X[:, 0:1] + 3 * X[:, 1:2] + 1 + np.random.randn(100, 1) * 0.1

# 2. Definir a arquitetura da rede
layers = [
    DenseLayer(n_units=4, input_shape=(2,)),
    ReLU(),
    DenseLayer(n_units=1)
]

# 3. Inicializar a rede, propagar os shapes e inicializar os pesos
loss_fn = MSELoss()
net = NeuralNetwork(layers=layers, loss_function=loss_fn)

input_shape = (2,)
for layer in net.layers:
    layer.set_input_shape(input_shape)
    if hasattr(layer, 'initialize'):
        layer.initialize()
    input_shape = layer.output_shape()

# 4. Treinar a rede
print("A treinar a rede neuronal...")
net.fit(X, y, epochs=100, lr=0.01)

# 5. Testar previsões
predictions = net.predict(X)
print("Primeiras previsões:", predictions[:5].flatten())
print("Valores reais (y):", y[:5].flatten())