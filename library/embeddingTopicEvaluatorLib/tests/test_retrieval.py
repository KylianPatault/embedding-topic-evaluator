# Tests pour la métrique de retrieval

import json
from ..metrics.retrieval import retrieval
from ..models.bertopic_wrapper import TopicModelEvaluatorBERTopic
from .__init__ import TestTopicModelEvaluator

def test_retrieval_metrics():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)
    metrics = retrieval(topic_model=topic_model, docs=TestTopicModelEvaluator.SAMPLE_DOCS)
    assert isinstance(metrics, dict)
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    test_retrieval_metrics()
