# Topic Mixing Tests — génération de tâches Label Studio et calcul des scores

import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import TopicModelEvaluator
from ..utils.embeddings import calculCentroide


def _find_closest_topic(
    topic_key: int,
    keys: list[int],
    centroids: dict[int, np.ndarray],
) -> int:
    """
    Retourne l'identifiant du topic le plus proche de `topic_key`
    en termes de similarité cosinus entre leurs centroïdes.
    """
    ref = centroids[topic_key].reshape(1, -1)
    best_key, best_sim = None, -np.inf
    for k in keys:
        if k == topic_key:
            continue
        sim = cosine_similarity(ref, centroids[k].reshape(1, -1))[0, 0]
        if sim > best_sim:
            best_sim = sim
            best_key = k
    return best_key


def generate_tasks_multi(
    model: TopicModelEvaluator,
    n_words_per_topic: int = 5,
    useEmbeddingModel: bool = False,
) -> list[dict]:
    """
    Génère les tâches du Topic Mixing Test (identification multi-topic) pour Label Studio.

    Pour chaque topic :
    - Trouve le topic le plus proche (similarité cosinus entre centroïdes).
    - Prend les n_words_per_topic premiers mots de chaque topic.
    - Mélange les 2 * n_words_per_topic mots aléatoirement.

    Args:
        model             : Instance de TopicModelEvaluator entraîné.
        n_words_per_topic : Nombre de mots prélevés dans chacun des 2 topics.
        useEmbeddingModel : Si True, utilise le modèle d'embeddings pour les centroïdes.

    Retourne une liste de dicts importables directement dans Label Studio.
    Chaque dict contient :
        - topic_id_1 (int)  : identifiant du topic principal.
        - topic_id_2 (int)  : identifiant du topic le plus proche.
        - word_0 … word_N   : mots mélangés (N = 2 * n_words_per_topic - 1).
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]

    # Pré-calcul des centroïdes de chaque topic
    centroids: dict[int, np.ndarray] = {
        k: calculCentroide(
            word_topics=model.getTopicWords(k)[:n_words_per_topic],
            model=model,
            useEmbeddingModel=useEmbeddingModel,
        )
        for k in keys
    }

    tasks = []

    for topic_key in keys:
        closest_key = _find_closest_topic(topic_key, keys, centroids)

        words_a = model.getTopicWords(topic_key)[:n_words_per_topic]
        words_b = model.getTopicWords(closest_key)[:n_words_per_topic]

        all_words = words_a + words_b
        random.shuffle(all_words)

        task = {
            "topic_id_1": topic_key,
            "topic_id_2": closest_key,
        }
        for i, word in enumerate(all_words):
            task[f"word_{i}"] = word

        tasks.append(task)

    return tasks


def topic_mixing_score_multi(annotations: list[dict]) -> float:
    """
    Calcule le score du Multi-Topic Identification Test à partir des annotations
    exportées de Label Studio.

    Une tâche est réussie si l'annotateur a sélectionné exactement les 2 bons
    identifiants de topics (ordre non important).

    Args:
        annotations : liste exportée depuis Label Studio (format JSON).

    Retourne un float entre 0.0 et 1.0.
    """
    correct, total = 0, 0

    for task in annotations:
        data = task.get("data", {})
        gt_1 = data.get("topic_id_1")
        gt_2 = data.get("topic_id_2")

        if gt_1 is None or gt_2 is None:
            continue

        task_annotations = task.get("annotations", [])
        if not task_annotations:
            continue

        results = task_annotations[0].get("result", [])
        if not results:
            continue

        try:
            # Label Studio retourne les choix sous forme de liste de strings
            chosen = results[0]["value"]["choices"]
        except (KeyError, IndexError):
            continue

        total += 1
        chosen_set = {int(c) for c in chosen}
        ground_truth_set = {gt_1, gt_2}

        if chosen_set == ground_truth_set:
            correct += 1

    return correct / total if total > 0 else 0.0


def generate_tasks_single(
    model: TopicModelEvaluator,
    n_words: int = 10,
) -> list[dict]:
    """
    Génère les tâches du Topic Mixing Test (détection single topic) pour Label Studio.

    Pour chaque topic, génère une tâche présentant n_words mots issus d'un seul topic.
    L'annotateur doit répondre : « 1 topic » ou « Plusieurs topics ».

    Args:
        model   : Instance de TopicModelEvaluator entraîné.
        n_words : Nombre de mots présentés à l'annotateur.

    Retourne une liste de dicts importables directement dans Label Studio.
    Chaque dict contient :
        - topic_id_1 (int) : identifiant du topic source.
        - word_0 … word_N  : mots du topic (N = n_words - 1).
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]

    tasks = []

    for topic_key in keys:
        words = model.getTopicWords(topic_key)[:n_words]

        task = {"topic_id_1": topic_key}
        for i, word in enumerate(words):
            task[f"word_{i}"] = word

        tasks.append(task)

    return tasks


def topic_mixing_score_single(annotations: list[dict]) -> float:
    """
    Calcule le score du Single-Topic Detection Test à partir des annotations
    exportées de Label Studio.

    Une tâche est réussie si l'annotateur répond « 1 topic »
    (toutes les tâches sont à topic unique).

    Args:
        annotations : liste exportée depuis Label Studio (format JSON).

    Retourne un float entre 0.0 et 1.0.
    """
    correct, total = 0, 0

    for task in annotations:
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
        if chosen == "1 topic":
            correct += 1

    return correct / total if total > 0 else 0.0


def save_tasks(tasks: list[dict], path: str) -> None:
    """Sauvegarde les tâches au format JSON pour import dans Label Studio."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
