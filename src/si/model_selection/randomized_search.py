import numpy as np
from si.model_selection.split import k_fold_cross_validation

def log_uniform(low: float, high: float):
    """
    Retorna uma função que amostra valores uniformemente numa escala logarítmica entre low e high.
    """
    def sampler(rng: np.random.Generator):
        log_low = np.log10(low)
        log_high = np.log10(high)
        return 10 ** rng.uniform(log_low, log_high)
    return sampler

def randomized_search_cv(model, dataset, hyperparameter_distributions, scoring=None, cv=5, n_iter=10, seed=None):
    """
    Realiza uma pesquisa aleatória (Randomized Search) com validação cruzada k-fold.
    
    Parameters
    ----------
    model : callable or object
        O modelo a ser sintonizado ou uma função construtora (builder).
    dataset : Dataset
        O dataset de treino.
    hyperparameter_distributions : dict
        Dicionário com os nomes dos hiperparâmetros e listas de valores ou funções de amostragem.
    scoring : callable, optional
        Função de pontuação. Se None, utiliza model.score.
    cv : int, default=5
        Número de folds para a validação cruzada.
    n_iter : int, default=10
        Número de configurações aleatórias a avaliar.
    seed : int, optional
        Semente para a aleatoriedade.
        
    Returns
    -------
    dict
        Dicionário contendo os hiperparâmetros testados, as pontuações médias,
        e a melhor configuração encontrada.
    """
    hyperparameters = []
    scores = []
    
    # Criar o gerador de números aleatórios com a seed fornecida
    rng = np.random.default_rng(seed)
    
    for _ in range(n_iter):
        config = {}
        for name, dist in hyperparameter_distributions.items():
            if callable(dist):
                # Se for uma função (ex: log_uniform), chamamos passando o rng
                config[name] = dist(rng)
            elif isinstance(dist, list):
                # Se for uma lista, escolhemos um valor aleatoriamente
                config[name] = rng.choice(dist)
            else:
                raise ValueError(f"Distribuição inválida para o hiperparâmetro {name}")
                
        hyperparameters.append(config)
        
        # Construir o modelo com a configuração atual
        if callable(model):
            current_model = model(**config)
        else:
            current_model = model
            for key, value in config.items():
                setattr(current_model, key, value)
                
        # Executar a validação cruzada k-fold
        fold_scores = k_fold_cross_validation(current_model, dataset, cv=cv)
        mean_score = np.mean(fold_scores)
        scores.append(mean_score)
        
    # Encontrar a melhor configuração
    best_index = np.argmax(scores)
    best_hyperparameters = hyperparameters[best_index]
    best_score = scores[best_index]
    
    return {
        'hyperparameters': hyperparameters,
        'scores': scores,
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }