# Métrique de diversité des topics

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic

from ..utils.topics import getWordsTopics
from ..utils.embeddings import calculCentroide
def diversity(model :BERTopic)-> float :
    keys = model.get_topics().keys()

    centroides = []
    for key in keys :
        words= getWordsTopics(key,model)
        centroides.append(calculCentroide(words,model))
    
    arrayCentroides = np.array(centroides)

    sim_matrix = cosine_similarity(arrayCentroides)
    np.fill_diagonal(sim_matrix, -1)
    res = sim_matrix.max(axis=1) 
    return np.mean(res)