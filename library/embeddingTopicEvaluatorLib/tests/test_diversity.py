# Tests pour la métrique de diversité

import numpy as np
from ..metrics.diversity import diversity
from ..models.bertopic_wrapper import TopicModelEvaluatorBERTopic
from .__init__ import TestTopicModelEvaluator

def test_diversity_metrics():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)
    metrics = diversity(model=topic_model)
    assert isinstance(metrics, np.float32)
    print(metrics)

if __name__ == '__main__':
    test_diversity_metrics()
