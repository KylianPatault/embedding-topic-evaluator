# Métrique de retrieval des topics

import numpy as np
import pytrec_eval
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import TopicModelEvaluator

def retrieval(topic_model: TopicModelEvaluator, docs: list) -> {}:
    """
    Calcule les métriques de "retrieval" en préparant le dictionnaire qrel et run à l'aide de la bibliothèque pytrec_eval.
    Métriques : 
        - Mean Average Precision (MAP)
        - Normalized Discounted Cumulative Gain (NDCG)
    topic_model: modèle TopicModelEvaluator
    return: dictionnaire des métriques
    """
    
    qrel, predictions = prepare_data(topic_model=topic_model, docs=docs)

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
    results = evaluator.evaluate(predictions)
    
    return results

def prepare_data(topic_model: TopicModelEvaluator, docs: list) -> tuple[dict, dict]:
    """
    Prépare le dictionnaire qrel et run pour la bibliothèque pytrec_eval
    topic_model: modèle TopicModelEvaluator
    return: dictionnaire qrel et run
    """
    
    documentInfos = topic_model.getDocumentInfos(docs)
    documentInfos = documentInfos[documentInfos['Topic'] != -1]

    # Faire l'embedding de tous les documents d'un coup
    all_docs = documentInfos['Document'].tolist()
    docs_embeddings = np.array(
        topic_model.getDocumentsVectors(all_docs)
    ) # shape: (n_docs, embedding_dim)

    doc_ids = [str(doc) for doc in documentInfos.index]
    doc_topics = documentInfos['Topic'].tolist()
    
    topicsKeys = topic_model.getTopicsKeys()
    topics = {k: topic_model.getTopicWords(k) for k in topicsKeys if k != -1}
    qrel = dict()
    predictions = dict()

    for topic, words_topic in topics.items():
        topic_id = str(topic)
        qrel[topic_id] = {}
        predictions[topic_id] = {}

        # Faire l'embedding des mots-clés
        embeddingMotsCles = topic_model.getWordVectors(
            words=[w for w in words_topic]
        )
        
        # Calcul des scores
        scores = cosine_similarity(docs_embeddings, embeddingMotsCles).mean(axis=1)

        # Construction des dicts
        predictions[topic_id] = {
            doc_id: float(score)
            for doc_id, score in zip(doc_ids, scores)
        }
        qrel[topic_id] = {
            doc_id: 1 if t == topic else 0
            for doc_id, t in zip(doc_ids, doc_topics)
        }
        
    return qrel, predictions