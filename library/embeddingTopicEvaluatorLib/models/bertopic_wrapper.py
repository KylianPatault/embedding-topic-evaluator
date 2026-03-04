# Wrapper pour le modèle BERTopic
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from ..config.config import settings


def load_model_BERTopic(config : dict = None) -> BERTopic:
    """
    Cette fonction permet de charger un modèle BERTopic à partir d'un fichier de configuration.
    Si le fichier de configuration n'est pas fourni, la configuration par défaut est chargée (Situé dans le fichier config/config.py).
    La fonction retourne le modèle BERTopic charger. 
    """
    
    # Vérification de la présence ou non d'un fichier de configuration 
    if config is None :
        config = settings.BERTOPIC_CONFIG_HDBSCAN
    umap_model = None 
    hdbscan_model = None 

    # Vérification de la présence ou non d'un fichier de configuration pour UMAP, sinon défini à None. 
    if "UMAP" in config.keys() : 
        umap_config = config["UMAP"]
        umap_model = UMAP(n_neighbors=umap_config["n_neighbors"], n_components=umap_config["n_components"], 
                          min_dist=umap_config["min_dist"], metric=umap_config["metric"])
        
    # Vérification de la présence ou non d'un fichier de configuration pour HDBSCAN, sinon défini à None. 
    if "HDBSCAN" in config.keys() : 
        hdbscan_config = config["HDBSCAN"]
        hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_config["min_cluster_size"], min_samples=hdbscan_config["min_samples"] , 
                                metric=hdbscan_config["metric"], cluster_selection_method=hdbscan_config["cluster_selection_method"], 
                                prediction_data=hdbscan_config["prediction_data"])

    # Vérification de la présence ou non d'un fichier de configuration pour KMeans, si HDBSCAN non défini, sinon défini à None. 
    elif "KMeans" in config.keys() :
        hdbscan_config = config["KMeans"]
        hdbscan_model = KMeans(n_clusters=hdbscan_config["n_clusters"]) 

    # Création du modèle BERTopic  
    berTopic_config =  config["BERTopic"]
    topic_model = BERTopic(embedding_model=berTopic_config["embedding_model"], 
                           umap_model=umap_model, hdbscan_model=hdbscan_model, 
                           nr_topics=berTopic_config["nr_topics"], verbose=berTopic_config["verbose"])
    return topic_model