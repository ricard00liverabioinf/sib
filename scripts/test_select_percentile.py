from si.io.csv_file import read_csv
from si.feature_selection.select_percentile import SelectPercentile

# 1. Carregar o iris.csv
dataset = read_csv('datasets/iris.csv', sep=',', features=True, label=True)
print(f"Dataset original: {dataset.X.shape[1]} features ({dataset.features})")

# 2. Aplicar o SelectPercentile para selecionar 50% das melhores features
selector = SelectPercentile(percentile=50)
selector.fit(dataset)
new_dataset = selector.transform(dataset)

print(f"F-values: {selector.F}")
print(f"P-values: {selector.p}")
print(f"Dataset reduzido: {new_dataset.X.shape[1]} features ({new_dataset.features})")