import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(dataset.X)
    permutations = np.random.permutation(n_samples)
    test_samples = int(n_samples * test_size)
    
    test_idxs = permutations[:test_samples]
    train_idxs = permutations[test_samples:]
    
    train_dataset = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], dataset.features, dataset.label)
    test_dataset = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], dataset.features, dataset.label)
    
    return train_dataset, test_dataset