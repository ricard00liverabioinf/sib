import numpy as np
from si.data.dataset import Dataset

def read_data_file(filename: str, sep: str = ",", label: bool = True) -> Dataset:
    """
    Lê um ficheiro de dados e devolve um objeto Dataset.
    """
    data = np.genfromtxt(filename, delimiter=sep)
    if label:
        X = data[:, :-1]
        y = data[:, -1]
    else:
        X = data
        y = None
    return Dataset(X=X, y=y)

def write_data_file(filename: str, dataset: Dataset, sep: str = ",", label: bool = True):
    """
    Escreve um objeto Dataset num ficheiro.
    """
    if label and dataset.has_label():
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X
    np.savetxt(filename, data, delimiter=sep)