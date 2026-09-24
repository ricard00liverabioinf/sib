import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    df = pd.read_csv(filename, sep=sep)
    
    if features and label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feat_names = df.columns[:-1].tolist()
        lbl_name = df.columns[-1]
    elif features and not label:
        X = df.to_numpy()
        y = None
        feat_names = df.columns.tolist()
        lbl_name = None
    elif not features and label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feat_names = None
        lbl_name = None
    else:
        X = df.to_numpy()
        y = None
        feat_names = None
        lbl_name = None

    return Dataset(X=X, y=y, features=feat_names, label=lbl_name)

def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    data = pd.DataFrame(dataset.X, columns=dataset.features if features and dataset.has_features() else None)
    
    if label and dataset.has_label():
        data[dataset.label] = dataset.y
        
    data.to_csv(filename, sep=sep, index=False)