class DefaultSettings():
    """Configuration par défaut des modèles"""

    BERTOPIC_CONFIG = {
        "UMAP" : {
            "n_neighbors" : 30,
            "n_components" : 10,
            "min_dist" : 0.0,
            "metric" : "cosine"
        },
        "HDBSCAN" : {
            "min_cluster_size" : 50,
            "min_samples" : 10,
            "metric" :"euclidean",
            "cluster_selection_method" : "eom",
            "prediction_data" : True
        },
        "BERTopic" : {
            "embedding_model" : "all-mpnet-base-v2",
            "nr_topics" : "auto",
            "verbose" : True
        }
    }

settings = DefaultSettings()