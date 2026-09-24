import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    def __init__(self, score_func: Callable = f_classification, percentile: float = 50.0):
        super().__init__()
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        n_features = dataset.X.shape[1]
        k = int(np.ceil(n_features * (self.percentile / 100.0)))
        
        threshold = np.percentile(self.F, 100.0 - self.percentile)
        mask = self.F > threshold
        
        if np.sum(mask) < k:
            tied_indices = np.where(self.F == threshold)[0]
            needed = k - np.sum(mask)
            mask[tied_indices[:needed]] = True
            
        new_X = dataset.X[:, mask]
        new_feature_names = np.array(dataset.features)[mask] if dataset.features else None
        
        return Dataset(
            X=new_X,
            y=dataset.y,
            features=list(new_feature_names) if new_feature_names is not None else None,
            label=dataset.label
        )