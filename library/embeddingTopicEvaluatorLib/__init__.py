# embedding_topic_evaluator
# Plateforme d'évaluation de topic models (BERTopic, Top2Vec)
# Métriques : cohérence, retrieval, diversité

from .config import config 
from .models import bertopic_wrapper, base
from .utils import embeddings, topics