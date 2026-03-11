# Word Intrusion Test — génération de tâches Label Studio et calcul du score

import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import TopicModelEvaluator
from ..utils.embeddings import calculCentroide


def generate_tasks(model: TopicModelEvaluator, n_words: int = 5) -> list[dict]:
    """
    Génère les tâches du Word Intrusion Test pour Label Studio.

    Pour chaque topic :
    - Prend les n_words premiers mots du topic.
    - Sélectionne un mot intrus : le mot des autres topics le moins similaire
      au centroïde du topic cible.
    - Mélange les n_words + 1 mots aléatoirement.

    Retourne une liste de dicts importables directement dans Label Studio.
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]
    all_topics = {k: model.getTopicWords(k)[:n_words] for k in keys}
    tasks = []

    for topic_key in keys:
        target_words = all_topics[topic_key]

        # Centroïde du topic cible
        centroid = calculCentroide(word_topics=target_words, model=model)

        # Candidats : top mots de tous les autres topics, absents du topic cible
        target_set = set(target_words)
        candidates = [
            w for k, words in all_topics.items()
            if k != topic_key               # Uniquement les autres topics
            for w in words
            if w not in target_set          # Uniquement les mots n'appartenant pas au topic cible
        ]

        # Choix du mot le moins similaire au centroïde cible
        candidate_vectors = model.getWordVectors(candidates)
        similarities = cosine_similarity(candidate_vectors, centroid).flatten()
        intruder = candidates[int(np.argmin(similarities))]

        # Mélange et encodage
        all_words = target_words + [intruder]
        random.shuffle(all_words)

        task = {"topic_id": topic_key, "intruder": intruder}
        for i, word in enumerate(all_words):
            task[f"word_{i}"] = word

        tasks.append(task)

    return tasks


def save_tasks(tasks: list[dict], path: str) -> None:
    """Sauvegarde les tâches au format JSON pour import dans Label Studio."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def word_intrusion_score(annotations: list[dict]) -> float:
    """
    Calcule le Model Precision à partir des annotations exportées de Label Studio.

    annotations : liste exportée depuis Label Studio (format JSON).
    Retourne un float entre 0.0 et 1.0.
    """
    correct, total = 0, 0

    for task in annotations:
        ground_truth = task.get("data", {}).get("intruder")
        task_annotations = task.get("annotations", [])
        if not task_annotations:
            continue
        results = task_annotations[0].get("result", [])
        if not results:
            continue
        try:
            chosen = results[0]["value"]["choices"][0]
        except (KeyError, IndexError):
            continue
        total += 1
        if chosen == ground_truth:
            correct += 1

    return correct / total if total > 0 else 0.0
