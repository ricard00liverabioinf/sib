import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from si.data.dataset import Dataset
from si.neural_networks.layers import DenseLayer, SigmoidActivation
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import MeanSquaredError

# Métrica de accuracy simples
def accuracy(y_true, y_pred):
    return np.mean((y_true == np.round(y_pred)))

# 1. Criar o dataset XOR (Slide 14/16)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)
ds = Dataset(X, y)

# 2. Configurar a rede neuronal
np.random.seed(1)
net = NeuralNetwork(
    epochs=5000, 
    batch_size=4, 
    optimizer=SGD, 
    learning_rate=0.5, 
    loss=MeanSquaredError(), 
    metric=accuracy,
    verbose=True
)

net.add(DenseLayer(4, input_shape=(2,)))
net.add(SigmoidActivation())
net.add(DenseLayer(1))
net.add(SigmoidActivation())

# Verificar os shapes de cada camada antes do treino
print("--- Verificação da Arquitetura ---")
input_shape = ds.X.shape[1:]
for i, layer in enumerate(net.layers):
    layer.set_input_shape(input_shape)
    if hasattr(layer, 'initialize'):
        layer.initialize()
    print(f"Camada {i} ({layer.__class__.__name__}): input_shape = {layer.input_shape()}, output_shape = {layer.output_shape()}")
    input_shape = layer.output_shape()
print("----------------------------------\n")

# 3. Treinar a rede
print("A treinar a rede no problema XOR...")
net.fit(ds)

# 4. Testar previsões
print("\nPrevisões finais (arredondadas):")
print(net.predict(ds).round(2))