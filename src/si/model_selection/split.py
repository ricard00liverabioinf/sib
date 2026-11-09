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

def k_fold_cross_validation(model, dataset: Dataset, cv: int = 3):
    """
    Realiza a validação cruzada k-fold para um modelo num dado dataset.
    """
    n_samples = len(dataset.X)
    fold_size = n_samples // cv
    scores = []
    
    # Baralhar os índices para garantir aleatoriedade
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in range(cv):
        # Separar índices de treino e validação para este fold
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = np.setdiff1d(indices, val_indices)
        
        # Criar sub-datasets mantendo features e label se existirem
        train_dataset = Dataset(dataset.X[train_indices], dataset.y[train_indices], dataset.features, dataset.label)
        val_dataset = Dataset(dataset.X[val_indices], dataset.y[val_indices], dataset.features, dataset.label)
        
        # Treinar o modelo e calcular a pontuação
        model.fit(train_dataset)
        score = model.score(val_dataset)
        scores.append(score)
        
    return np.array(scores)