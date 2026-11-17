import itertools
import numpy as np
from si.model_selection.split import k_fold_cross_validation

def grid_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, seed=None):
    """
    Realiza uma pesquisa em grelha (Grid Search) com validação cruzada k-fold.
    
    Parameters
    ----------
    model : callable or object
        O modelo a ser sintonizado ou uma função construtora (builder).
    dataset : Dataset
        O dataset de treino.
    hyperparameter_grid : dict
        Dicionário com os nomes dos hiperparâmetros e listas de valores a testar.
    scoring : callable, optional
        Função de pontuação. Se None, utiliza model.score.
    cv : int, default=5
        Número de folds para a validação cruzada.
    seed : int, optional
        Semente para a aleatoriedade dos folds.
        
    Returns
    -------
    dict
        Dicionário contendo os hiperparâmetros testados, as pontuações médias,
        e a melhor configuração encontrada.
    """
    hyperparameters = []
    scores = []
    
    # Extrair chaves e valores da grelha de hiperparâmetros
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    
    # Gerar todas as combinações possíveis de hiperparâmetros (itertools.product)
    for combination in itertools.product(*values):
        # Criar um dicionário para a configuração atual
        config = dict(zip(keys, combination))
        hyperparameters.append(config)
        
        # Se for fornecida uma função construtora (builder), invocamos com os parâmetros da config
        if callable(model):
            # Nota: se o modelo precisar de seed ou parâmetros específicos, passamos o dicionário desdobrado
            current_model = model(**config)
        else:
            # Se for um objeto modelo, atualizamos os atributos (ex: copy.deepcopy + setattr se necessário)
            # Para redes neuronais, o recomendado é usar a função construtora (builder)
            current_model = model
            for key, value in config.items():
                setattr(current_model, key, value)
                
        # Executar a validação cruzada k-fold para esta configuração
        # (Nota: a k_fold_cross_validation existente pode ser adaptada ou chamada diretamente)
        fold_scores = k_fold_cross_validation(current_model, dataset, cv=cv)
        
        # Calcular a pontuação média dos folds
        mean_score = np.mean(fold_scores)
        scores.append(mean_score)
        
    # Encontrar a melhor configuração com base na pontuação média mais alta
    best_index = np.argmax(scores)
    best_hyperparameters = hyperparameters[best_index]
    best_score = scores[best_index]
    
    return {
        'hyperparameters': hyperparameters,
        'scores': scores,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }