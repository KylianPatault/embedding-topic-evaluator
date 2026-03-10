# Classe de base abstraite pour les wrappers de modèles de topics
from typing import List
import numpy as np
import pandas as pd

class TopicModelEvaluator:
    """
    Classe abstraite représentant un modèle de production de topics  
        
    Attributs :
    config (dict) : la configuration du modèle     
    """
    def __init__(self, config : dict = None):
        self.model = None

    def getWordVectors(self, words: list) -> np.ndarray:
        """
        Récupère les embeddings pour une liste de mots donnés.
        """
        return np.array()

    def getDocumentsVectors(self, documents: list) -> np.ndarray:
        """
        Récupère les embeddings pour une liste de documents donnés.
        """
        return np.array()

    def getTopicWords(self, topic_key: int) -> List[str]:
        """
        Extrait uniquement les mots (sans les poids) pour n'importe quel modèle.
        """
        return []

    def getTopicsKeys(self) -> list :
        """
        Retourne les clés des topics.
        """
        return []

    def evaluate(self, docs):
        """
        Entraîne le modèle et retourne les topics et les probabilités.
        """
        return None, None

    def getDocumentInfos(self, docs: list) -> pd.DataFrame:
        """
        Retourne toutes les informations des documents
        dans un dataframe Pandas avec les colonnes suivantes :
            - Document (str) : Le text original du document.
            - Topic (int) : L'identifiant du topic assigné. 
            - Name (str) : Nom descriptif du topic.
            - Probability (float) : Score de confiance/proximité (cosinus) entre le document et le centre de son topic.
            - Representative_Doc (bool) : Indicateur booléen. True si le document  est l'un des plus centraux du cluster.
        """
        return None