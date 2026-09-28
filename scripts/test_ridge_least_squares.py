from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

# 1. Carregar dataset
dataset = read_csv('datasets/cpu/cpu.csv', sep=',', features=True, label=True)

# 2. Dividir em treino e teste
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Treinar modelo de Mínimos Quadrados
model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
model.fit(train)

# 4. Avaliar
score = model.score(test)
print(f"MSE no Teste (Least Squares): {score:.4f}")