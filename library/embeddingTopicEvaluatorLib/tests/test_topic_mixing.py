# Tests pour les Topic Mixing Tests

from ..human_evaluation.topic_mixing import (
    generate_tasks_multi,
    generate_tasks_single,
    topic_mixing_score_multi,
    topic_mixing_score_single,
)
from ..models.bertopic_wrapper import TopicModelEvaluatorBERTopic
from .__init__ import TestTopicModelEvaluator


def test_generate_tasks_multi():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)

    n_words_per_topic = 3
    tasks = generate_tasks_multi(topic_model, n_words_per_topic=n_words_per_topic)

    assert isinstance(tasks, list)
    assert len(tasks) > 0

    for task in tasks:
        assert "topic_id_1" in task
        assert "topic_id_2" in task
        # Les deux topics doivent être différents
        assert task["topic_id_1"] != task["topic_id_2"]
        # Doit contenir des mots (au moins word_0)
        assert "word_0" in task
        # Le nombre total de mots doit être 2 × n_words_per_topic
        word_keys = [k for k in task if k.startswith("word_")]
        assert len(word_keys) == 2 * n_words_per_topic

    print(f"generate_tasks_multi: {len(tasks)} tâches générées")
    print(tasks[0])


def test_generate_tasks_single():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)

    n_words = 6
    tasks = generate_tasks_single(topic_model, n_words=n_words)

    assert isinstance(tasks, list)
    assert len(tasks) > 0

    for task in tasks:
        assert "topic_id_1" in task
        word_keys = [k for k in task if k.startswith("word_")]
        assert len(word_keys) == n_words

    print(f"generate_tasks_single: {len(tasks)} tâches générées")
    print(tasks[0])


def test_topic_mixing_score_multi():
    # Simule des annotations Label Studio pour le test multi-topic
    mock_annotations = [
        {
            "data": {"topic_id_1": 0, "topic_id_2": 1},
            "annotations": [{"result": [{"value": {"choices": ["0", "1"]}}]}],
        },
        {
            "data": {"topic_id_1": 0, "topic_id_2": 2},
            "annotations": [{"result": [{"value": {"choices": ["0", "3"]}}]}],  # mauvais
        },
        {
            "data": {"topic_id_1": 1, "topic_id_2": 2},
            "annotations": [{"result": [{"value": {"choices": ["2", "1"]}}]}],  # ordre inversé OK
        },
    ]
    score = topic_mixing_score_multi(mock_annotations)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # 2 correctes sur 3
    assert abs(score - 2 / 3) < 1e-9
    print(f"topic_mixing_score_multi: {score:.4f}")


def test_topic_mixing_score_single():
    # Simule des annotations Label Studio pour le test single-topic
    # Toutes les tâches sont à 1 seul topic → la bonne réponse est toujours « 1 topic »
    mock_annotations = [
        {
            "data": {"topic_id_1": 0},
            "annotations": [{"result": [{"value": {"choices": ["1 topic"]}}]}],  # correct
        },
        {
            "data": {"topic_id_1": 1},
            "annotations": [{"result": [{"value": {"choices": ["1 topic"]}}]}],  # correct
        },
        {
            "data": {"topic_id_1": 2},
            "annotations": [{"result": [{"value": {"choices": ["Plusieurs topics"]}}]}],  # mauvais
        },
    ]
    score = topic_mixing_score_single(mock_annotations)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # 2 correctes sur 3
    assert abs(score - 2 / 3) < 1e-9
    print(f"topic_mixing_score_single: {score:.4f}")


if __name__ == "__main__":
    test_generate_tasks_multi()
    test_generate_tasks_single()
    test_topic_mixing_score_multi()
    test_topic_mixing_score_single()
