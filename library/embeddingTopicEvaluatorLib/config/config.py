class DefaultSettings():
    """Configuration par défaut des modèles"""

    BERTOPIC_CONFIG_HDBSCAN = {
        "UMAP" : {
            "n_neighbors" : 15, # Taille du voisinage local (plus élevé = vision plus globale)
            "n_components" : 5, # Dimension de sortie pour le clustering
            "min_dist" : 0.0, # Distance min entre les points
            "metric" : "cosine" # Métrique de distance
        },
        "HDBSCAN" : {
            "min_cluster_size" : 50, # Taille min d'un topic
            "min_samples" : 10, # Sensibilité au bruit (plus bas = moins d'exclus)
            "metric" :"euclidean", # Distance standard après UMAP
            "cluster_selection_method" : "eom", # Sélection des clusters les plus denses
            "prediction_data" : True # Permet de classer de nouveaux documents
        },
        "BERTopic" : {
            # Modèle de Sentence Transformers utilisé pour les embeddings
            "embedding_model" : "all-mpnet-base-v2",
            "nr_topics" : "auto", # Réduction automatique du nombre de topics si nécessaire
            "verbose" : True # Affiche la barre de progression
        }
    }

    BERTOPIC_CONFIG_KMeans = {
        "UMAP" : {
            "n_neighbors" : 15, # Taille du voisinage local (plus élevé = vision plus globale)
            "n_components" : 5, # Dimension de sortie pour le clustering
            "min_dist" : 0.0, # Distance min entre les points
            "metric" : "cosine" # Métrique de distance
        },
        "KMeans" : {
            "n_clusters" : 50, # Nombre de thèmes (topics) ciblés
        },
        "BERTopic" : {
            # Modèle de Sentence Transformers utilisé pour les embeddings
            "embedding_model" : "all-mpnet-base-v2",
            "nr_topics" : "auto", # Réduction automatique du nombre de topics si nécessaire
            "verbose" : True # Affiche la barre de progression
        }
    }
settings = DefaultSettings()