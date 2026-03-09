# Wrapper pour le modèle BERTopic
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import numpy as np
from typing import List
import pandas as pd

from ..config.config import settings
from .base import TopicModelEvaluator

def load_model_BERTopic(config : dict = None) -> BERTopic:
    """
    Cette fonction permet de charger un modèle BERTopic à partir d'un fichier de configuration.
    Si le fichier de configuration n'est pas fourni, la configuration par défaut est chargée (située dans le fichier config/config.py).
    La fonction retourne le modèle BERTopic chargé. 
    """
    
    # Vérification de la présence ou non d'un fichier de configuration 
    if config is None :
        config = settings.BERTOPIC_CONFIG_HDBSCAN
    umap_model = None 
    hdbscan_model = None 

    # Vérification de la présence ou non d'un fichier de configuration pour UMAP, si non défini à None. 
    if "UMAP" in config.keys() : 
        if config["UMAP"] is not None :
            umap_config = config["UMAP"]
            umap_model = UMAP(n_neighbors=umap_config["n_neighbors"], n_components=umap_config["n_components"], 
                              min_dist=umap_config["min_dist"], metric=umap_config["metric"])
        
    # Vérification de la présence ou non d'un fichier de configuration pour HDBSCAN, si non défini à None. 
    if "HDBSCAN" in config.keys() : 
        if config["HDBSCAN"] is not None :
            hdbscan_config = config["HDBSCAN"]
            hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_config["min_cluster_size"], min_samples=hdbscan_config["min_samples"] , 
                                    metric=hdbscan_config["metric"], cluster_selection_method=hdbscan_config["cluster_selection_method"], 
                                    prediction_data=hdbscan_config["prediction_data"])

    # Vérification de la présence ou non d'un fichier de configuration pour KMeans, si HDBSCAN non défini, sinon défini à None. 
    elif "KMeans" in config.keys() :
        if config["KMeans"] is not None :
            hdbscan_config = config["KMeans"]
            hdbscan_model = KMeans(n_clusters=hdbscan_config["n_clusters"]) 

    # Création du modèle BERTopic
    berTopic_config = config["BERTopic"]
    topic_model = BERTopic(embedding_model=berTopic_config["embedding_model"], 
                           umap_model=umap_model, hdbscan_model=hdbscan_model, 
                           nr_topics=berTopic_config["nr_topics"], verbose=berTopic_config["verbose"])
    return topic_model

class TopicModelEvaluatorBERTopic(TopicModelEvaluator):
    """
    Classe dériver de la classe TopicModelEvaluator 
    Elle représentant un modèle de production de topics de type BERTopic
        
    Attributs :
    config (dict) : la configuration du modèle     
    """
    def __init__(self, config : dict = None):
        super().__init__(config)
        self.model = load_model_BERTopic(config)

    def getWordVectors(self, words: list) -> np.ndarray:
        """Récupère les embeddings pour une liste de mots donnés."""
        return self.model.embedding_model.embed_words(words)

    def getDocumentsVectors(self, documents: list) -> np.ndarray:
        """
        Récupère les embeddings pour une liste de documents donnés.
        """
        return self.model.embedding_model.embed_documents(documents)
        
    def getTopicWords(self, topic_key: int) -> List[str]:
        """Extrait uniquement les mots (sans les poids) pour n'importe quel modèle."""
        topic_info = self.model.get_topic(topic_key)
        if not topic_info: return []
                
        return [word for word, _ in topic_info]

    def getTopicsKeys(self) -> list :
        """Retourne les clés des topics"""
        return self.model.get_topics().keys()
            
    def evaluate(self, docs):
        """
        Entraîne le modèle et retourne les topics et les probabilités.
        """
        topics, probs=self.model.fit_transform(docs)
        return topics, probs

    def getDocumentInfos(self, docs: list) -> pd.DataFrame:
        """
        Retourne toutes les informations des documents
        """
        return self.model.get_document_info(docs)