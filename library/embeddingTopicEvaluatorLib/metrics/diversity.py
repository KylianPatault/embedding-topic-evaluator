# Métrique de diversité des topics

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from typing import Callable

from ..utils.embeddings import calculCentroide
from ..models.base import TopicModelEvaluator

def diversity(model :TopicModelEvaluator, distance : Callable[[np.ndarray], np.ndarray] = cosine_similarity, maximise : bool = True )-> float :
    """
    Cette fonction permet de calcule la diversité moyenne entre tous les centroides des topics, mais en fonction de leurs mots de référence. 
    Elle a besoin du modèle de production de topic qui est encapsuler dans la classe TopicModelEvaluator, de comment elle doit comparer les centroides de base, c'est une Cosine Similarity et de définir si on doit maximiser ou non cette comparaison. 
    """

    keys = [k for k in model.getTopicsKeys() if k != -1]
    centroides = []
    for key in keys :
        words = model.getTopicWords(key)
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