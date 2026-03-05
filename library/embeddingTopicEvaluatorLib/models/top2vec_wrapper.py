# Wrapper pour le modèle Top2Vec
from top2vec import Top2Vec

from ..config.config import settings


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
    topic_model = Top2Vec(config["embedding_model"], 
                            umap_model=umap_model, 
                            min_count=config["min_count"], 
                            nr_topics=config["nr_topics"], 
                            verbose=config["verbose"])
    return topic_model