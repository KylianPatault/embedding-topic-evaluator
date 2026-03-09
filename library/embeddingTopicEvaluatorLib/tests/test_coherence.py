# Tests pour la métrique de cohérence

from ..metrics.coherence import coherence
from ..models.bertopic_wrapper import TopicModelEvaluatorBERTopic
from .__init__ import TestTopicModelEvaluator

def test_coherence_metrics():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)
    metrics = coherence(model=topic_model)
    assert isinstance(metrics, dict)
    print(metrics)

if __name__ == '__main__':
    test_coherence_metrics()
