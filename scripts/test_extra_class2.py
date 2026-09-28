import numpy as np
from si.data.dataset import Dataset
from si.statistics.manhattan_distance import manhattan_distance
from si.statistics.minkowski_distance import minkowski_distance
from si.models.logistic_regression import LogisticRegression

print("--- 1. Testar Novas Distâncias (Manhattan e Minkowski) ---")
x = np.array([1, 2, 3])
y = np.array([[4, 5, 6], [1, 2, 3]])

print("Distância de Manhattan:", manhattan_distance(x, y))
print("Distância de Minkowski (p=3):", minkowski_distance(x, y, p=3))

print("\n--- 2. Testar Logistic Regression (L1 e Momentum) ---")
# Criar um dataset sintético de classificação binária
X_dummy = np.random.randn(100, 4)
y_dummy = np.random.choice([0, 1], size=100)
dataset = Dataset(X=X_dummy, y=y_dummy)

# Testar com Momentum e L1
model_l1 = LogisticRegression(penalty='l1', l1_penalty=0.5, alpha=0.01, momentum=0.9, max_iter=200)
model_l1.fit(dataset)
acc = model_l1.score(dataset)

print(f"Treino concluído com sucesso!")
print(f"Accuracy no dataset de teste: {acc:.4f}")
print(f"Custo inicial: {model_l1.cost_history[0]:.4f}")
print(f"Custo final: {model_l1.cost_history[199]:.4f}")