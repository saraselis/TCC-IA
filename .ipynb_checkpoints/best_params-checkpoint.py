import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RED = "\033[1;31m"
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
PINK = "\033[1;45m"
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
MAG = "\033[1;45m"


def best_params(values: dict, clf: 'classificador', np_df, n_iter_) -> list:
    
    logger.info('Instanciando Kmeans')
    random_clf = RandomizedSearchCV(clf, param_distributions=values, n_iter=n_iter_, verbose=1)
    
    logger.info('Treinando Kmeans')
    random_clf.fit(np_df)
    
    logger.info('Parametros Kmeans')
    print(RED, random_clf.get_params())
    
    logger.info('Best Params Kmeans')
    best_params = random_clf.best_params_
    print(BLUE, best_params)
    
    return best_params, random_clf