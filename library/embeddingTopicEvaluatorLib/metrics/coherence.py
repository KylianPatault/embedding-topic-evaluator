from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic

from ..models.base import TopicModelEvaluator

def coherence(model: TopicModelEvaluator, useEmbeddingModel: bool = False) -> dict:
    """
    Calcule la cohérence sémantique de chaque topic du modèle.

    La cohérence mesure à quel point les mots d'un même topic sont
    sémantiquement proches les uns des autres. Pour cela, on calcule
    la similarité cosinus entre les embeddings de chaque paire de mots
    du topic, puis on en fait la moyenne.

    Un score élevé (proche de 1) indique que les mots du topic sont
    fortement liés sémantiquement, donc que le topic est cohérent.
    Un score faible indique que les mots sont peu liés entre eux.

    Args:
        model: Instance de TopicModelEvaluator fournissant les topics
               et les embeddings des mots.

    Returns:
        dict: Dictionnaire {topic_id: score_de_cohérence} où chaque
              score est un float entre -1 et 1 (moyenne des similarités
              cosinus entre toutes les paires de mots du topic).
    """
    # Récupération des identifiants de tous les topics
    keys = model.getTopicsKeys()
    res = dict()
    
    for key in keys:
        # Pour chaque topic, on récupère ses mots et leurs vecteurs (embeddings)
        words = model.getTopicWords(key)
        word_embeddings = model.getWordVectors(words, useEmbeddingModel)

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