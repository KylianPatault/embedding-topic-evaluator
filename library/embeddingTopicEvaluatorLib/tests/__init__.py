class TestTopicModelEvaluator:
    """
    Classe de test pour le modèle TopicModelEvaluator
    """
    
    TEST_CONFIG = {
        "UMAP": {
            "n_neighbors": 3,       # Doit être < nombre de documents
            "n_components": 2,      # Dimensionnalité réduite suffisante pour les tests
            "min_dist": 0.0,
            "metric": "cosine"
        },
        "HDBSCAN": {
            "min_cluster_size": 2,  # Permet de créer des topics avec peu de documents
            "min_samples": 1,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        },
        "BERTopic": {
            "embedding_model": "all-mpnet-base-v2",
            "nr_topics": "auto",
            "verbose": False        # Pas de barre de progression pendant les tests
        }
    }

    SAMPLE_DOCS = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Transfer learning reuses models trained on other tasks.",
        "Neural networks are inspired by the human brain structure.",
        "Data preprocessing is a crucial step in machine learning pipelines.",
    ]
