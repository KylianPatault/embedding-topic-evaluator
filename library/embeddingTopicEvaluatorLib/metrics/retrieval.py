# Métrique de retrieval des topics

from bertopic import BERTopic
import numpy as np
import pytrec_eval

class RetrievalMetrics:
    def __init__(self, target: np.ndarray, probs: np.ndarray):
        """
        target: tableau des étiquettes réelles
        probs: tableau des probabilités
        """
        self.target = target
        self.probs = probs

    def calculate_retrieval_metrics(self) -> dict:
        """
        Calcule les métriques de "retrieval" en préparant les fichiers qrel et run à l'aide de la bibliothèque pytrec_eval.
        Métriques : 
            - Mean Average Precision (MAP)
            - Normalized Discounted Cumulative Gain (NDCG)
            - Precision at 5 (P@5)
            - Reciprocal Rank (RR)
        return: dictionnaire des métriques
        """
        metrics = {}
        qrel = self.prepare_qrel(self.target)
        run = self.prepare_run(self.probs)
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg', 'P_5', 'recip_rank'})
        metrics = evaluator.evaluate(run)
        return metrics
    
    def prepare_qrel(self, true_labels: np.ndarray) -> dict:
        """
        Prépare le fichier qrel pour la bibliothèque pytrec_eval
        true_labels: tableau des étiquettes réelles
        return: dictionnaire qrel
        """
        qrel = {}
        for doc_idx, label in enumerate(true_labels):
            topic_id = str(label)
            doc_id = str(doc_idx)
            if topic_id not in qrel:
                qrel[doc_id] = {}
            qrel[topic_id][doc_id] = 1
        return qrel

    def prepare_run(self, probs: np.ndarray) -> dict:
        """
        Prépare le fichier run pour la bibliothèque pytrec_eval
        probs: tableau des probabilités
        return: dictionnaire run
        """
        run = {}
        for topic_idx in range(probs.shape[1]):
            topic_id = str(topic_idx)
            run[topic_id] = {}
            for doc_idx in range(probs.shape[0]):
                doc_id = str(doc_idx)
                run[topic_id][doc_id] = float(probs[doc_idx, topic_idx])
        return run