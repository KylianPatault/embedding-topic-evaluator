# Wrapper pour le modèle Top2Vec
import numpy as np
from typing import List

from .base import TopicModelEvaluator

class TopicModelEvaluatorTop2Vec(TopicModelEvaluator):
    """
    Classe dériver de la classe TopicModelEvaluator 
    Elle représentant un modèle de production de topics de type Top2Vec
        
    Attributs :
    config (dict) : la configuration du modèle    
    
    """
    def __init__(self, config : dict = None):
        super().__init__(config)
        self.model = load_model_BERTopic(config)

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