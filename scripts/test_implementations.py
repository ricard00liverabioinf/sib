import numpy as np
from si.io.csv_file import read_csv, write_csv
from si.metrics.accuracy import accuracy
from si.statistics.f_classification import f_classification

# 1. Testar IO e Dataset
dataset = read_csv('datasets/iris.csv', sep=',', features=True, label=True)
print(f"Dataset carregado: {dataset.X.shape[0]} amostras, {dataset.X.shape[1]} features")

# 2. Testar Metrics (Accuracy)
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])
acc = accuracy(y_true, y_pred)
print(f"Accuracy de teste: {acc:.2f}")

# 3. Testar Statistics (F-Classification)
f_vals, p_vals = f_classification(dataset)
print("F-values:", f_vals)
print("P-values:", p_vals)