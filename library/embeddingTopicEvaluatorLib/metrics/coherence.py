from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic

from ..models.base import TopicModelEvaluator

# Cette fonction permet de calculer la cohérence des topics
def coherence(model: TopicModelEvaluator) -> dict:
    # Récupération des identifiants de tous les topics
    keys = model.getTopicsKeys()
    res = dict()
    
    for key in keys:
        # Pour chaque topic, on récupère ses mots et leurs vecteurs (embeddings)
        words = model.getTopicWords(key)
        word_embeddings = model.getWordVectors(words)

        # Calcul de la matrice de similarité cosinus entre tous les mots du topic
        sim_matrix = cosine_similarity(word_embeddings)

        n = sim_matrix.shape[0]
        # S'il y a moins de 2 mots, impossible de comparer : la cohérence est 0
        if n < 2:
            res[key] = np.float32(0.0)
            continue
            
        # On extrait le triangle supérieur de la matrice (k=1 exclut la diagonale)
        # Pour éviter de compter la similarité d'un mot avec lui-même et les doublons (A-B et B-A)
        upper_triangle = sim_matrix[np.triu_indices(n, k=1)]
        
        # La cohérence du topic est la moyenne de ces similarités
        res[key] = np.mean(upper_triangle)

    return res