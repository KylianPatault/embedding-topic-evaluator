# Wrapper pour le modèle Top2Vec
from top2vec import Top2Vec
from umap import UMAP
import numpy as np
from typing import List
import pandas as pd 
from ..config.config import settings
from .base import TopicModelEvaluator

def load_model_Top2Vec(documents: list, config: dict = None) -> Top2Vec:
    """
    Cette fonction permet de charger un modèle Top2Vec à partir d'un fichier de configuration.
    Si le fichier de configuration n'est pas fourni, la configuration par défaut est chargée (située dans le fichier config/config.py).
    La fonction retourne le modèle Top2Vec chargé. 
    """
    
    # Vérification de la présence ou non d'un fichier de configuration 
    if config is None : 
        config = settings.TOP2VEC_CONFIG
    umap_model = None 
    
    # Vérification de la présence ou non d'un fichier de configuration pour UMAP, si non défini à None. 
    if "UMAP" in config.keys() : 
        if config["UMAP"] is not None : 
            umap_model = config["UMAP"]
    
    # Création du modèle Top2Vec
    top2vec_config = config["TOP2VEC"]
    topic_model = Top2Vec(embedding_model = top2vec_config["embedding_model"],
                          documents = documents,
                          umap_args = umap_model,
                          min_count = top2vec_config["min_count"],  
                          verbose = top2vec_config["verbose"])
    
    target_topics = top2vec_config["nr_topics"]
    if target_topics and target_topics > 0:
        try:
            topic_model.hierarchical_topic_reduction(num_topics=target_topics)
        except Exception as e:
            print(f"Erreur lors de la réduction des topics : {e}")
    return topic_model

class TopicModelEvaluatorTop2Vec(TopicModelEvaluator):
    """
    Classe dériver de la classe TopicModelEvaluator 
    Elle représentant un modèle de production de topics de type Top2Vec
        
    Attributs :
    config (dict) : la configuration du modèle    
    
    """
    def __init__(self, documents, config: dict = None):
        super().__init__(config)
        self.model = load_model_Top2Vec(documents, config)
        self.has_reduction = hasattr(self.model, 'hierarchy')
        if config is None :
            self.n_components = 10
        else :
            self.n_components = config["TOP2VEC"]["n_components"]

            if "EmbeddingModel" in config.keys():
                self.embeddingModel = config["EmbeddingModel"]
            

    def getWordVectors(self, words: list, useEmbeddingModel: bool = True) -> np.ndarray:
        """Récupère les embeddings pour une liste de mots donnés."""
        if self.embeddingModel is not None and useEmbeddingModel :
            return self.embeddingModel.encode(words)
        else:
            return np.array([self.model.word_vectors[self.model.word_indexes[w]] 
                     for w in words if w in self.model.word_indexes])
        
    def getDocumentsVectors(self, documents: list = None, useEmbeddingModel: bool = True) -> np.ndarray:
        """
        Récupère les embeddings pour les documents.
        Si documents est None, retourne les vecteurs du modèle entraîné.
        """
        if documents is None:
            return self.model.document_vectors
            
        if self.embeddingModel is not None and useEmbeddingModel :
            return self.embeddingModel.encode(documents)
        else:
            # Pour de nouveaux documents, Top2Vec utilise sa propre méthode
            return self.model.embed(documents)

    
    def getTopicWords(self, topic_key: int) -> List[str]:
        """Extrait uniquement les mots (sans les poids) pour n'importe quel modèle."""
        words, _, _ = self.model.get_topics(reduced=self.has_reduction)
        
        topic_words = list(words[topic_key])
        
        return topic_words[:self.n_components]
        

    def getTopicsKeys(self) -> list :
        """Retourne les clés des topics"""
        
        num_topics = self.model.get_num_topics(reduced=self.has_reduction)
        return list(range(num_topics))
            
    def evaluate(self, docs):
        """
        Entraîne le modèle et retourne les topics et les probabilités.
        """
        if self.has_reduction:
            # Si on a réduit, on doit utiliser cette méthode pour avoir les nouveaux labels
            topic_labels, topic_scores, _, _ = self.model.get_documents_topics(
                doc_ids=self.model.document_ids,
                reduced=True
            )
        else:
            # Sinon, on prend les attributs bruts (très rapide)
            topic_labels = self.model.doc_top
            topic_scores = self.model.doc_dist
            
        return topic_labels, topic_scores
        
    def getDocumentInfos(self, docs: list) -> pd.DataFrame:
        """
        Retourne toutes les informations des documents
        """
        topic_labels, topic_scores = self.evaluate(docs)
        
        # Récupérer les noms de topics
        num_topics = len(self.getTopicsKeys())
        topic_names = []
        for i in range(num_topics):
            words = self.getTopicWords(i) # On prend les 4 premiers mots
            topic_names.append(f"{i}_" + "_".join(words))
        
        # Mapping des noms pour chaque document
        # On gère le cas -1 (même si rare en Top2Vec)
        current_topic_names = [
            topic_names[label] if label != -1 else "Outlier" 
            for label in topic_labels
        ]

        # Construire le DataFrame au format exact de BERTopic
        df = pd.DataFrame({
            "Document": docs,
            "Topic": topic_labels,
            "Name": current_topic_names,
            "Probability": topic_scores,
            "Representative_Doc": False 
        })
        
        return df

