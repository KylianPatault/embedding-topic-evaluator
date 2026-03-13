# Évaluation humaine des topic models — Word Intrusion & Topic Mixing Tests

from .word_intrusion import (
    generate_tasks as generate_word_intrusion_tasks,
    save_tasks as save_word_intrusion_tasks,
    word_intrusion_score,
)

from .topic_mixing import (
    generate_tasks_mixed,
    save_tasks as save_topic_mixing_tasks,
    topic_mixing_score_mixed,
)
