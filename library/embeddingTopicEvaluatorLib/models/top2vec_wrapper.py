# Wrapper pour le modèle Top2Vec
from top2vec import Top2Vec
from umap import UMAP
import numpy as np
from typing import List

from ..config.config import settings
from .base import TopicModelEvaluator

def load_model_Top2Vec(config : dict = None) -> Top2Vec:
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
            umap_config = config["UMAP"]
            umap_model = UMAP(n_neighbors=umap_config["n_neighbors"], n_components=umap_config["n_components"], 
                              min_dist=umap_config["min_dist"], metric=umap_config["metric"])
    
    # Création du modèle Top2Vec
    top2vec_config = config["TOP2VEC"]
    topic_model = Top2Vec(embedding_model=top2vec_config["embedding_model"], 
                          umap_model=umap_model, 
                          min_count=top2vec_config["min_count"], 
                          nr_topics=top2vec_config["nr_topics"], 
                          verbose=top2vec_config["verbose"])
    return topic_model

class TopicModelEvaluatorTop2Vec(TopicModelEvaluator):
    """
    Classe dériver de la classe TopicModelEvaluator 
    Elle représentant un modèle de production de topics de type Top2Vec
        
    Attributs :
    config (dict) : la configuration du modèle    
    
    """
    def __init__(self, config : dict = None):
        super().__init__(config)
        self.model = load_model_Top2Vec(config)

    def getWordVectors(self, words: list) -> np.ndarray:
        """Récupère les embeddings pour une liste de mots donnés."""
        return np.array([self.model.word_vectors[self.model.word_dict[w]] 
                            for w in words if w in self.model.word_dict])
        
    def getTopicWords(self, topic_key: int) -> List[str]:
        """Extrait uniquement les mots (sans les poids) pour n'importe quel modèle."""
        return list(self.model.topic_words[topic_key])
        

    def getTopicsKeys(self) -> list :
        """Retourne les clés des topics"""
        
        num_topics = self.model.get_num_topics()
        return list(range(num_topics))
            
    def evaluate(self, docs):
        """
        Entraîne le modèle et retourne les topics et les probabilités.
        """
        
        return None, None

