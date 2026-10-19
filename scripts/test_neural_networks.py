import numpy as np
from si.neural_networks.layers import DenseLayer

# Definir uma semente para os números aleatórios para garantir reprodutibilidade
np.random.seed(42)

# Inicializar a camada densa: 4 neurónios, input de shape (3,)
layer = DenseLayer(n_units=4, input_shape=(3,)).initialize()

# Criar dados de entrada aleatórios (10 amostras, 3 features)
X = np.random.rand(10, 3)

# Executar forward propagation
out = layer.forward_propagation(X)

# Validar com asserts
assert out.shape == (10, 4), f"Esperado shape (10, 4), obtido {out.shape}"
assert layer.parameters() == 3 * 4 + 4, f"Esperado 16 parâmetros, obtido {layer.parameters()}"
assert layer.output_shape() == (4,), f"Esperado output shape (4,), obtido {layer.output_shape()}"

print("Todos os testes da DenseLayer passaram com sucesso!")
print(f"Shape da saída: {out.shape}")
print(f"Número de parâmetros: {layer.parameters()}")