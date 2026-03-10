from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Callable

from ..utils.embeddings import calculCentroide
from ..models.base import TopicModelEvaluator

def cohesion(model: TopicModelEvaluator, distance: Callable[[np.ndarray], np.ndarray] = cosine_similarity):
    """
    Cette fonction permet de calculer la cohésion entre les centroïdes des topics en fonction de leurs mots de référence et une phrase composer des mots qui définissent les topics. 
    Le but de la diversity est d'être maximisé. 
    La fonction requiert une instance de TopicModelEvaluator (encapsulant le modèle de topics), une métrique de comparaison des centroïdes (par défaut cosine_similarity).
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]
    centroides = []
    sentences = []
    for key in keys :
        words = model.getTopicWords(key)
        topic_words = " ".join(words)
        sentences.append(model.getDocumentsVectors(topic_words,False))
        centroides.append(calculCentroide(words, model))
        
    arrayCentroides = np.array(centroides)
    arraySentences = np.array(sentences)
    matrix = distance(arrayCentroides, arraySentences)

    return np.diag(matrix)