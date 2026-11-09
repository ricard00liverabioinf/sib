import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from si.data.dataset import Dataset
from si.neural_networks.layers import DenseLayer, ReLU, SigmoidActivation
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import BinaryCrossEntropy
from si.metrics.accuracy import accuracy
from si.model_selection.split import k_fold_cross_validation

# 1. Gerar um dataset sintético para classificação binária com 32 features
np.random.seed(42)
X = np.random.randn(200, 32)
y = np.random.randint(0, 2, size=200)
dataset = Dataset(X, y)

# 2. Definir a função modelo para o k-fold cross-validation
def evaluate_nn():
    net = NeuralNetwork(
        epochs=100,
        batch_size=16,
        optimizer=SGD,
        learning_rate=0.01,
        loss=BinaryCrossEntropy(),
        metric=accuracy,
        verbose=True
    )
    # Camadas com redução para metade (32 -> 16 -> 8 -> 1)
    net.add(DenseLayer(16, input_shape=(32,)))
    net.add(ReLU())
    net.add(DenseLayer(8))
    net.add(ReLU())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net

print("A executar a validação cruzada k-fold para o Exercício 16...")

# 3. Executar K-Fold Cross Validation (ex: k=5)
scores = k_fold_cross_validation(evaluate_nn(), dataset, cv=5)

print("\nResultados da Validação Cruzada (Accuracy por fold):")
print(scores)
print(f"Accuracy média: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")