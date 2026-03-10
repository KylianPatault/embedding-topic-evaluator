# Métrique de diversité des topics

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from typing import Callable

from ..utils.embeddings import calculCentroide
from ..models.base import TopicModelEvaluator

def diversity(model: TopicModelEvaluator, distance: Callable[[np.ndarray], np.ndarray] = cosine_similarity, 
              maximise: bool = True, useEmbeddingModel: bool = True )-> float :
    """
    Cette fonction permet de calculer la diversité moyenne entre tous les centroides des topics en fonction de leurs mots de référence. 
    La fonction requiert une instance de TopicModelEvaluator (encapsulant le modèle de topics), une métrique de comparaison des centroïdes (par défaut cosine_similarity), ainsi qu'un booléen définissant l'objectif d'optimisation (maximisation ou minimisation).
    """

    keys = [k for k in model.getTopicsKeys() if k != -1]
    centroides = []
    for key in keys :
        words = model.getTopicWords(key)
        if useEmbeddingModel :
            topic_words = " ".join(words)
            centroides.append(model.getDocumentsVectors(topic_words,useEmbeddingModel))
        else:
            centroides.append(calculCentroide(words, model))
    
    arrayCentroides = np.array(centroides)

    matrix = distance(arrayCentroides)
    
    if maximise:
        np.fill_diagonal(matrix, -np.inf)
        res = matrix.max(axis=1)
    else:
        np.fill_diagonal(matrix, np.inf)
        res = matrix.min(axis=1)
        
    return np.mean(res)