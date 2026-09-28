from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression import RidgeRegression

# 1. Carregar dataset cpu.csv
dataset = read_csv('datasets/cpu/cpu.csv', sep=',', features=True, label=True)

# 2. Divisão treino/teste
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Treinar modelo RidgeRegression
model = RidgeRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000)
model.fit(train)

print(f"Custo final no Teste: {model.cost(test):.4f}")