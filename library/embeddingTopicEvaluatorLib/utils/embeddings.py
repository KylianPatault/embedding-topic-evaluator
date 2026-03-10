# Fonctions utilitaires pour la gestion des embeddings

from bertopic import BERTopic
import numpy as np
from ..models.base import TopicModelEvaluator

def calculCentroide(word_topics :list, model :TopicModelEvaluator) -> float :
    """
    Cette fonction permet de calculer le centroide ("le vecteur moyen") pour les mots d'un topic donnée en paramètre.
    Pour ce faire elle a besoin des mots qui définissent le topic et du modèle de production de topic qui est encapsuler dans la classe TopicModelEvaluator.
    """
    word_embeddings = model.getWordVectors(word_topics)
    return np.mean(word_embeddings,axis=0)