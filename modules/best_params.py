import logging
import numpy as np
import sys

from sklearn.model_selection import RandomizedSearchCV


logger = logging.getLogger("LogReg")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s - %(levelname)s - [+] ------- %(message)s -------') 
handler.setFormatter(formatter)
logger.handlers = [handler]


def best_params(values: dict, clf: 'classificador', x_treino: np.array, x_teste: np.array, y_treino: np.array, y_teste: np.array) -> list:
    '''
        Instancia um classificador de busca e procura os melhores parâmetros para o modelo.
        
        Params
        ------
        :values: parametros a serem testados
        :clf: instancia do classificador desejado
        :x_treino: dados de treino
        :x_teste: dados de teste
        :y_treino: dados de treino -labels
        :y_teste: dados de treino - labels
        
        Return
        ------
        :best_params: lista com os melhores parametros
    '''
    
    logger.info('Instanciando Kmeans')
    random_clf = RandomizedSearchCV(clf, param_distributions=values, n_iter=200, verbose=10)
    
    logger.info('Treinando Kmeans')
    random_clf.fit(x_treino, y_treino)
    
    logger.info('Predict Kmeans')
    y_random_clf_rl = random_clf.predict(x_teste)
    #print(y_random_clf_rl)
    
    logger.info('Parametros Kmeans')
    print(RED, random_clf.get_params())
    
    logger.info('Best Params Kmeans')
    best_params = random_clf.best_params_
    print(BLUE, best_params)
    
    return best_params