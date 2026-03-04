# Métrique de retrieval des topics

from bertopic import BERTopic
import numpy as np
import pytrec_eval

def calculate_retrieval_metrics(target: np.ndarray, probs: np.ndarray) -> dict:
    """
    Calcule les métriques de "retrieval" en préparant le dictionnaire qrel et run à l'aide de la bibliothèque pytrec_eval.
    Métriques : 
        - Mean Average Precision (MAP)
        - Normalized Discounted Cumulative Gain (NDCG)
        - Precision at 5 (P@5)
        - Reciprocal Rank (RR)
    return: dictionnaire des métriques
    """
    metrics = {}
    qrel = prepare_qrel(target)
    run = prepare_run(probs)
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'P_5', 'recip_rank'})
    metrics = evaluator.evaluate(run) # Compare le run au qrel et renvoie les scores par topic
    return metrics

def prepare_qrel(true_labels: np.ndarray) -> dict:
    """
    Prépare le dictionnaire qrel pour la bibliothèque pytrec_eval
    true_labels: tableau des étiquettes réelles
    return: dictionnaire qrel
    """
    qrel = {}
    for doc_idx, label in enumerate(true_labels):
        topic_id = str(label)
        doc_id = str(doc_idx)
        if topic_id not in qrel:
            qrel[topic_id] = {}
        qrel[topic_id][doc_id] = 1 # Document pertinent pour le topic (score à 1)
    return qrel

def prepare_run(probs: np.ndarray) -> dict:
    """
    Prépare le dictionnaire run pour la bibliothèque pytrec_eval
    probs: tableau des probabilités
    return: dictionnaire run
    """
    run = {}
    for topic_idx in range(probs.shape[1]): # Pour chaque topic
        topic_id = str(topic_idx)
        run[topic_id] = {}
        for doc_idx in range(probs.shape[0]): # Pour chaque document
            doc_id = str(doc_idx)
            # Récupère la probabilité que le document doc_idx appartienne au topic topic_idx
            run[topic_id][doc_id] = float(probs[doc_idx, topic_idx])
    return run