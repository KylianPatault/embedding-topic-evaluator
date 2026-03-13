# Tests pour les Topic Mixing Tests

from ..human_evaluation.topic_mixing import (
    generate_tasks_mixed,
    topic_mixing_score_mixed,
)
from ..models.bertopic_wrapper import TopicModelEvaluatorBERTopic
from .__init__ import TestTopicModelEvaluator


def test_generate_tasks_mixed():
    topic_model = TopicModelEvaluatorBERTopic(config=TestTopicModelEvaluator.TEST_CONFIG)
    topic_model.evaluate(TestTopicModelEvaluator.SAMPLE_DOCS)

    n_words = 6
    tasks = generate_tasks_mixed(topic_model, n_words=n_words)

    assert isinstance(tasks, list)
    assert len(tasks) > 0

    topics_count = len([k for k in topic_model.getTopicsKeys() if k != -1])
    assert len(tasks) == topics_count * 2  # 1 single + 1 multi per topic

    single_count = 0
    multi_count = 0

    for task in tasks:
        assert "task_type" in task
        
        word_keys = [k for k in task if k.startswith("word_")]
        assert len(word_keys) == n_words

        if task["task_type"] == "single":
            single_count += 1
            assert "topic_id_1" in task
            assert "topic_id_2" not in task
        elif task["task_type"] == "multi":
            multi_count += 1
            assert "topic_id_1" in task
            assert "topic_id_2" in task
            assert task["topic_id_1"] != task["topic_id_2"]

    assert single_count == topics_count
    assert multi_count == topics_count

    print(f"generate_tasks_mixed: {len(tasks)} tâches générées ({single_count} single, {multi_count} multi)")


def test_topic_mixing_score_mixed():
    # Simule des annotations Label Studio pour le test unifié
    mock_annotations = [
        # --- Single Topic Tasks ---
        {
            "data": {"task_type": "single", "topic_id_1": 0},
            "annotations": [{"result": [{"value": {"choices": ["1 topic"]}}]}],  # correct
        },
        {
            "data": {"task_type": "single", "topic_id_1": 1},
            "annotations": [{"result": [{"value": {"choices": ["Plusieurs topics"]}}]}],  # mauvais
        },
        # --- Multi Topic Tasks ---
        {
            "data": {"task_type": "multi", "topic_id_1": 0, "topic_id_2": 1},
            "annotations": [{"result": [{"value": {"choices": ["0", "1"]}}]}],  # correct
        },
        {
            "data": {"task_type": "multi", "topic_id_1": 0, "topic_id_2": 2},
            "annotations": [{"result": [{"value": {"choices": ["0", "3"]}}]}],  # mauvais (mauvais ID)
        },
        {
            "data": {"task_type": "multi", "topic_id_1": 1, "topic_id_2": 2},
            "annotations": [{"result": [{"value": {"choices": ["2", "1"]}}]}],  # correct (ordre inversé)
        },
        {
            "data": {"task_type": "multi", "topic_id_1": 3, "topic_id_2": 4},
            "annotations": [{"result": [{"value": {"choices": ["1 topic"]}}]}],  # mauvais (a coché 1 topic au lieu des IDs)
        },
    ]

    scores = topic_mixing_score_mixed(mock_annotations)
    
    assert isinstance(scores, dict)
    assert 0.0 <= scores["score_global"] <= 1.0
    assert 0.0 <= scores["score_single"] <= 1.0
    assert 0.0 <= scores["score_multi"] <= 1.0

    # Single : 1 correct sur 2
    assert abs(scores["score_single"] - 0.5) < 1e-9
    
    # Multi : 2 corrects sur 4
    assert abs(scores["score_multi"] - 0.5) < 1e-9
    
    # Global : 3 corrects sur 6
    assert abs(scores["score_global"] - 0.5) < 1e-9

    assert scores["details"]["single_correct"] == 1
    assert scores["details"]["single_total"] == 2
    assert scores["details"]["multi_correct"] == 2
    assert scores["details"]["multi_total"] == 4

    print(f"topic_mixing_score_mixed: {scores}")


if __name__ == "__main__":
    test_generate_tasks_mixed()
    test_topic_mixing_score_mixed()
