# Évaluation de modèles d'embeddings de topic

Une plateforme d'évaluation pour les topic models neuronaux (BERTopic, Top2Vec) basée sur 3 métriques : cohérence, retrieval et diversité.

## Table des matières

- [Installation](#installation)
- [Modèles](#modèles)
  - [Top2Vec](#top2vec)
  - [BERTopic](#bertopic)
- [Métriques d'évaluation](#métriques-dévaluation)
  - [Cohérence](#cohérence)
  - [Retrieval](#retrieval)
  - [Diversité](#diversité)
- [Datasets](#datasets)
  - [20 Newsgroups](#20-newsgroups)
  - [AG News](#ag-news)
  - [ArXiv](#arxiv)
  - [Big Patent](#big-patent)
  - [BioRxiv](#biorxiv)
  - [ClusTREC-Covid](#clustrec-covid)
- [Résultats](#résultats)

## Installation

```bash
pip install -r requirements.txt
```

## Modèles

Ce projet se concentre sur l'évaluation de deux modèles d'embeddings pour comprendre la signification sémantique des documents.

### Top2Vec

Top2Vec est un algorithme de modélisation thématique et de recherche sémantique. Il détecte automatiquement les thèmes présents dans le texte et génère des vecteurs de thèmes, de documents et de mots représentés conjointement dans un même espace vectoriel.

### BERTopic

BERTopic est un framework moderne de modélisation thématique qui répond à de nombreuses limites des approches traditionnelles. Développé par Maarten Grootendorst , il utilise des plongements (embeddings) basés sur des transformeurs (comme BERT) pour comprendre la signification sémantique des documents et les regrouper en clusters en fonction de leur contexte, plutôt que de se baser uniquement sur la fréquence des mots.

## Métriques d'évaluation

Ce projet utilise trois métriques d'évaluation pour évaluer les performances des modèles. Ces métriques sont la cohérence, le retrieval et la diversité.

### Cohérence

Cette métrique calcule la similirité cosine entre le top K des mots définissant un topic afin de comprendre si les meilleurs mots représentant un topic sont cohérents entre eux.

### Retrieval

Cette métrique calcule la fidélité aux documents du topic en utilisant la méthode TREC. Cette approche soumet des requêtes dont chacune d'elles nous retourne les documents correspondant au topic fourni. On utilise deux métriques pour évaluer le retrieval :

- MAP (Mean Average Precision) : C'est la moyenne des précisions moyennes pour chaque requête.
- NDCG (Normalized Discounted Cumulative Gain) : C'est la moyenne des gains cumulés normalisés pour chaque requête.

### Diversité

Cette méthode calcule la distance entre les centroïdes des topics les plus proches, puis on calcule la moyenne des sommes de ces distances.

## Datasets

### 20 Newsgroups

Le dataset 20 Newsgroups est un ensemble de données de 20 000 articles de presse répartis en 20 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### AG News

Le dataset AG News est un ensemble de données de 120 000 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### ArXiv

Le dataset ArXiv est un ensemble de données de 100 000 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### Big Patent

Le dataset Big Patent est un ensemble de données de 100 000 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### BioRxiv

Le dataset BioRxiv est un ensemble de données de 100 000 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### ClusTREC-Covid

Le dataset ClusTREC-Covid est un ensemble de données de 100 000 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

## Résultats

| Modèle | Dataset | Nombre de topics | Cohérence | Retrieval-MAP | Retrieval-NDCG | Diversité |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BERTopic | 20 Newsgroups | 20 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | 20 Newsgroups | 20 | 0.XX | 0.XX | 0.XX | 0.XX |
| BERTopic | AG News | 4 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | AG News | 4 | 0.XX | 0.XX | 0.XX | 0.XX |
| BERTopic | ArXiv | 50 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | ArXiv | 50 | 0.XX | 0.XX | 0.XX | 0.XX |
| BERTopic | Big Patent | 100 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | Big Patent | 100 | 0.XX | 0.XX | 0.XX | 0.XX |
| BERTopic | BioRxiv | 30 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | BioRxiv | 30 | 0.XX | 0.XX | 0.XX | 0.XX |
| BERTopic | ClusTREC-Covid | 10 | 0.XX | 0.XX | 0.XX | 0.XX |
| Top2Vec | ClusTREC-Covid | 10 | 0.XX | 0.XX | 0.XX | 0.XX |
