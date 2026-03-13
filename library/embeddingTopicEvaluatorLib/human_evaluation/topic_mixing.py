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


def generate_tasks_mixed(
    model: TopicModelEvaluator,
    n_words: int = 10,
    useEmbeddingModel: bool = False,
    stopWords: list[str] = [],
) -> list[dict]:
    """
    Génère les tâches unifiées du Topic Mixing Test pour Label Studio.

    Pour chaque topic valide, génère 2 tâches :
    1. Une tâche "Single Topic" : prend n_words mots du topic, l'annotateur doit cliquer sur "1 topic".
    2. Une tâche "Multi Topic" : trouve le topic le plus proche, prend n_words//2 de chaque, 
       les mélange, l'annotateur doit identifier les 2 topics d'origine.

    Args:
        model             : Instance de TopicModelEvaluator entraîné.
        n_words           : Nombre total de mots présentés dans chaque tâche. 
                            Pour les tâches Multi Topic, on tire n_words // 2 mots 
                            du topic principal, et le reste (n_words - n_words // 2) 
                            du topic le plus proche.
        useEmbeddingModel : Si True, utilise le modèle d'embeddings pour les centroïdes.
        stopWords         : Liste de mots à ne pas inclure dans les tâches.

    Retourne une liste de dicts mélangés aléatoirement, importables dans Label Studio.
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]

    # Pré-calcul des mots filtrés pour chaque topic
    all_topics = {k: [w for w in model.getTopicWords(k) if w not in stopWords] for k in keys}

    # Pré-calcul des centroïdes locaux sur les N premiers mots filtrés
    centroids: dict[int, np.ndarray] = {
        k: calculCentroide(
            word_topics=all_topics[k][:n_words],
            model=model,
            useEmbeddingModel=useEmbeddingModel,
        )
        for k in keys
    }

    tasks = []

    for topic_key in keys:
        # 1. Tâche Single Topic
        single_words = all_topics[topic_key][:n_words]
        if len(single_words) < n_words:
            continue
        task_single = {
            "task_type": "single",
            "topic_id_1": topic_key,
        }
        for i, word in enumerate(single_words):
            task_single[f"word_{i}"] = word
        tasks.append(task_single)

        # 2. Tâche Multi Topic
        closest_key = _find_closest_topic(topic_key, keys, centroids)
        n_first = n_words // 2
        n_second = n_words - n_first

        words_a = all_topics[topic_key][:n_first]
        words_b = all_topics[closest_key][:n_second]

        if len(words_a) < n_first or len(words_b) < n_second:
            continue

        multi_words = words_a + words_b
        random.shuffle(multi_words)

        task_multi = {
            "task_type": "multi",
            "topic_id_1": topic_key,
            "topic_id_2": closest_key,
        }
        for i, word in enumerate(multi_words):
            task_multi[f"word_{i}"] = word
        tasks.append(task_multi)

    # Mélanger l'ordre global des tâches pour éviter que l'annotateur
    # ne devine le pattern (single, puis multi, puis single...)
    random.shuffle(tasks)

    return tasks


def topic_mixing_score_mixed(annotations: list[dict]) -> dict:
    """
    Calcule les scores du Topic Mixing Test unifié à partir des annotations Label Studio.

    Il différencie les réponses selon le type de tâche (Single ou Multi).
    - Single : l'annotateur doit choisir uniquement "1 topic".
    - Multi : l'annotateur doit sélectionner les 2 identifiants exacts des topics.

    Args:
        annotations : liste exportée depuis Label Studio (format JSON).

    Retourne :
        Un dictionnaire contenant les scores détaillés:
        {
            "score_global": float,
            "score_single": float,
            "score_multi": float,
            "details": {"single_correct": x, "single_total": y, "multi_correct": w, "multi_total": z}
        }
    """
    single_correct, single_total = 0, 0
    multi_correct, multi_total = 0, 0

    for task in annotations:
        data = task.get("data", {})
        task_type = data.get("task_type")

        # Fallback pour rétablir une ancienne compatibilité ou ignorer des tâches mal formées
        if not task_type:
            if "topic_id_2" in data:
                task_type = "multi"
            else:
                task_type = "single"

        task_annotations = task.get("annotations", [])
        if not task_annotations:
            continue

        results = task_annotations[0].get("result", [])
        if not results:
            continue

        try:
            chosen = results[0]["value"]["choices"]
        except (KeyError, IndexError):
            continue

        if task_type == "single":
            single_total += 1
            # Pour un test single topic, on attend que l'utilisateur n'ait coché qu'une case : "1 topic"
            if len(chosen) == 1 and chosen[0] == "1 topic":
                single_correct += 1

        elif task_type == "multi":
            multi_total += 1
            gt_1 = data.get("topic_id_1")
            gt_2 = data.get("topic_id_2")

            if gt_1 is not None and gt_2 is not None:
                chosen_set = {int(c) for c in chosen if c.isdigit()}
                ground_truth_set = {gt_1, gt_2}

                if chosen_set == ground_truth_set:
                    multi_correct += 1

    total_correct = single_correct + multi_correct
    total_tasks = single_total + multi_total

    return {
        "score_global": total_correct / total_tasks if total_tasks > 0 else 0.0,
        "score_single": single_correct / single_total if single_total > 0 else 0.0,
        "score_multi": multi_correct / multi_total if multi_total > 0 else 0.0,
        "details": {
            "single_correct": single_correct,
            "single_total": single_total,
            "multi_correct": multi_correct,
            "multi_total": multi_total,
        }
    }


def save_tasks(tasks: list[dict], path: str) -> None:
    """Sauvegarde les tâches au format JSON pour import dans Label Studio."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
