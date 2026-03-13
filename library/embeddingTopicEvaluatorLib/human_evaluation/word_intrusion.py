# Word Intrusion Test — génération de tâches Label Studio et calcul du score

import random
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import TopicModelEvaluator
from ..utils.embeddings import calculCentroide


def generate_tasks(
    model: TopicModelEvaluator,
    n_words: int = 5,
    top_distant_pct: float = 0.2,
    top_active_pct: float = 0.2,
    useEmbeddingModel: bool = False,
    stopWords: list[str] = [],
) -> list[dict]:
    """
    Génère les tâches du Word Intrusion Test pour Label Studio.

    Pour chaque topic :
    - Prend les n_words premiers mots du topic.
    - Sélectionne un mot intrus parmi les candidats des autres topics,
      en appliquant deux filtres successifs :
        1. top_distant_pct : ne conserve que les X% de candidats les plus
           éloignés du centroïde du topic cible (peu similaires au topic cible).
        2. top_active_pct  : parmi ceux-ci, ne conserve que les X% de candidats
           les mieux classés dans leur propre topic (très représentatifs d'un
           autre topic).
    - Tire l'intrus au hasard parmi les candidats restants.
    - Mélange les n_words + 1 mots aléatoirement.

    Args:
        model           : Instance de TopicModelEvaluator entraîné.
        n_words         : Nombre de mots du topic à présenter (hors intrus).
        top_distant_pct : Fraction [0, 1] des candidats les plus éloignés du
                          centroïde cible à conserver. 1.0 = tous les candidats.
        top_active_pct  : Fraction [0, 1] des candidats les mieux classés dans
                          leur topic d'origine à conserver. 1.0 = tous.
        stopWords       : Liste de mots à ne pas inclure dans les tâches.

    Retourne une liste de dicts importables directement dans Label Studio.
    """
    keys = [k for k in model.getTopicsKeys() if k != -1]
    # Pré-calcul : top mots (filtrés) + leur rang (index dans la liste) pour chaque topic
    all_topics = {k: [w for w in model.getTopicWords(k) if w not in stopWords] for k in keys}
    tasks = []

    for topic_key in keys:
        target_words = all_topics[topic_key][:n_words]
        if len(target_words) < n_words:
            continue

        # Centroïde du topic cible
        centroid = calculCentroide(word_topics=target_words, model=model, useEmbeddingModel=useEmbeddingModel).reshape(1, -1)

        # Candidats : top mots des autres topics absents du topic cible
        # On mémorise aussi le rang du mot dans son propre topic (0 = le meilleur)
        target_set = set(target_words)
        candidates = []       # liste de mots
        candidate_ranks = []  # rang du mot dans son topic d'origine (plus petit = meilleur)

        for k, words in all_topics.items():
            if k == topic_key:
                continue
            for rank, word in enumerate(words):
                if word not in target_set:
                    candidates.append(word)
                    candidate_ranks.append(rank)

        if not candidates:
            continue

        candidate_vectors = model.getWordVectors(candidates)
        similarities = cosine_similarity(candidate_vectors, centroid).flatten()

        n = len(candidates)

        # np.ceil permet d'arrondir au supérieur
        k_distant = int(np.ceil(n * top_distant_pct))
        k_active = int(np.ceil(n * top_active_pct))

        # Les plus éloignés du centroïde cible (similarité faible → ordre croissant)
        les_plus_eloignes = top_k_indices(similarities, k_distant if k_distant > 0 else 1, ascending=True)
        # Les mieux classés dans leur topic d'origine (rang faible → ordre croissant)
        les_plus_actifs = top_k_indices(candidate_ranks, k_active if k_active > 0 else 1, ascending=True)

        # Intersection des deux ensembles
        candidats_retenus = list(les_plus_eloignes & les_plus_actifs)
        if not candidats_retenus:
            # Si aucun candidat retenu avec l'intersection, on élargit à l'union
            candidats_retenus = list(les_plus_eloignes | les_plus_actifs)

        # On choisit un intrus au hasard parmi les candidats retenus
        intruder = candidates[random.choice(candidats_retenus)]

        # Mélange et encodage
        all_words = target_words + [intruder]
        random.shuffle(all_words)

        task = {"topic_id": topic_key, "intruder": intruder}
        for i, word in enumerate(all_words):
            task[f"word_{i}"] = word

        tasks.append(task)

    return tasks


def top_k_indices(scores: np.ndarray, k: int, ascending: bool) -> set[int]:
    """Retourne les indices des k meilleurs scores (croissant ou décroissant)."""
    sorted_indices = np.argsort(scores) if ascending else np.argsort(scores)[::-1]
    return set(sorted_indices[:k].tolist())


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
