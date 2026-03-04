# Fonctions utilitaires pour la gestion des embeddings

from bertopic import BERTopic
import numpy as np

def calculCentroide(word_topics :list,model :BERTopic) -> float :
    word_embeddings = model.embedding_model.embed_words(word_topics)
    return np.mean(word_embeddings,axis=1)