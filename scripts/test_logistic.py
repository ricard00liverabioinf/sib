from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from si.models.logistic_regression import LogisticRegression

# 1. Carregar o dataset 'breast-bin.csv'
# (Certifica-te de que o ficheiro está na pasta 'datasets/' do teu projeto)
dataset = read_data_file('datasets/breast_bin/breast-bin.csv', sep=',', label=True)

# 2. Dividir o dataset em treino (80%) e teste (20%)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. Inicializar e treinar o modelo de Regressão Logística
model = LogisticRegression(l2_penalty=1.0, alpha=0.001, max_iter=1000)
model.fit(train_dataset)

# 4. Avaliar o modelo (Score / Precisão)
score = model.score(test_dataset)
print(f"Accuracy (Score) no conjunto de teste: {score * 100:.2f}%")

