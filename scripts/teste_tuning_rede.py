import numpy as np
from si.data.dataset import Dataset
from si.model_selection.split import train_test_split
from si.model_selection.randomized_search import randomized_search_cv, log_uniform
from si.metrics.accuracy import accuracy
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.layers import DenseLayer, ReLU, SigmoidActivation
from si.neural_networks.losses import BinaryCrossEntropy
from si.neural_networks.optimizers import SGD

# 1. Criar um dataset sintético de classificação binária (semelhante ao Exercício 16)
rng = np.random.default_rng(42)
X = rng.standard_normal(size=(300, 32))
y = rng.integers(0, 2, size=(300,))
dataset = Dataset(X, y)

# 2. Dividir os dados em treino e teste (pesquisa apenas no treino!)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# 3. Definir a função construtora (builder) da NeuralNetwork
def build_network(learning_rate, batch_size, epochs):
    # Fixar a seed dentro do builder garante pesos iniciais iguais para cada configuração
    np.random.seed(42)
    net = NeuralNetwork(
        epochs=epochs,
        batch_size=batch_size,
        optimizer=SGD,
        learning_rate=learning_rate,
        momentum=0.9,
        loss=BinaryCrossEntropy,
        metric=accuracy
    )
    net.add(DenseLayer(8, (32,)))
    net.add(ReLU())  # <--- Corrigido para ReLU (conforme definido em layers.py)
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())
    return net

# 4. Definir as distribuições de hiperparâmetros exigidas pelo Exercício 18
hyperparameter_distributions = {
    'learning_rate': log_uniform(1e-4, 1e-1),
    'batch_size': [8, 16, 32, 64],
    'epochs': [50, 100, 200]
}

print("A executar a pesquisa aleatória de hiperparâmetros...")

# 5. Executar o randomized_search_cv
results = randomized_search_cv(
    model=build_network,
    dataset=train_dataset,
    hyperparameter_distributions=hyperparameter_distributions,
    scoring=accuracy,
    cv=3,
    n_iter=20,
    seed=42
)

print("\n--- Resultados da Otimização ---")
print("Melhores hiperparâmetros:", results['best_hyperparameters'])
print("Melhor score de validação cruzada:", round(results['best_score'], 4))

# 6. Refit: Treinar a melhor configuração em todo o conjunto de treino e avaliar no teste
best_params = results['best_hyperparameters']
best_model = build_network(**best_params)

print("\nA treinar o melhor modelo no conjunto de treino completo...")
best_model.fit(train_dataset)

test_score = best_model.score(test_dataset)
print(f"Acurácia final no conjunto de teste: {test_score:.4f}")