# Évaluation de modèles d'embeddings de topic

Une plateforme d'évaluation pour les topic models neuronaux (BERTopic, Top2Vec) basée sur 3 métriques : cohérence, retrieval et diversité. Ainsi qu'un word intrusion test pour évaluer la qualité des topics générés.

## Table des matières

- [Installation](#installation)
- [Modèles](#modèles)
  - [Top2Vec](#top2vec)
  - [BERTopic](#bertopic)
- [Métriques d'évaluation](#métriques-dévaluation)
  - [Cohérence](#cohérence)
  - [Retrieval](#retrieval)
  - [Diversité](#diversité)
- [Word Intrusion Test](#word-intrusion-test)
  - [Choix des mots](#choix-des-mots)
  - [Choix de l'intrus](#choix-de-lintrus)
- [Architecture](#architecture)
  - [Structure des modules](#structure-des-modules)
  - [Choix d'architecture](#choix-darchitecture)
- [Datasets](#datasets)
  - [20 Newsgroups](#20-newsgroups)
  - [AG News](#ag-news)
  - [ArXiv](#arxiv)
  - [Big Patent](#big-patent)
  - [BioRxiv](#biorxiv)
  - [ClusTREC-Covid](#clustrec-covid)
- [Résultats](#résultats)
  - [Observations Cohérence](#observations-cohérence)
  - [Observations Retrieval](#observations-retrieval)
  - [Observations Diversité](#observations-diversité)

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

## Word Intrusion Test

Le Word Intrusion Test est une méthode d'évaluation qualitative des topic models. Il consiste à présenter à des annotateurs humains une liste de mots pour chaque topic, dont un mot est un intrus. L'annotateur doit identifier l'intrus. Le score est calculé comme le pourcentage d'intrus correctement identifiés.

### Choix des mots

Pour chaque topic, le modèle (BERTopic ou Top2Vec) fournit une liste ordonnée de mots caractéristiques, du plus représentatif au moins représentatif. On retient les `n_words` premiers (par défaut 5), qui correspondent aux mots ayant le score de pertinence le plus élevé pour ce topic.

Ces mots sont ensuite mélangés aléatoirement avec l'intrus avant d'être présentés à l'annotateur, afin d'éviter tout biais de position.

### Choix de l'intrus

On compare chaque candidat au centroïde du topic cible, et on choisit celui
qui a la similarité cosinus la plus faible, c'est-à-dire le mot :

- le plus éloigné du centroïde
- le plus susceptible d'être un intrus

L'intrus n'est pas choisi au hasard : c'est le mot qui maximise l'incohérence
avec le topic cible tout en étant légitime dans un autre contexte. Cela rend
le test plus exigeant : si l'annotateur trouve quand même l'intrus, c'est
que le topic est vraiment cohérent.

## Architecture

Le projet est organisé en une librairie Python installable (`embeddingTopicEvaluatorLib`) dont la structure reflète une séparation claire des responsabilités.

### Structure des modules

```text
embeddingTopicEvaluatorLib/
├── models/          # Wrappers des modèles de topics
│   ├── base.py      # Classe de base abstraite TopicModelEvaluator
│   ├── bertopic_wrapper.py
│   └── top2vec_wrapper.py
├── metrics/         # Métriques d'évaluation (fonctions pures)
│   ├── coherence.py
│   ├── diversity.py
│   └── retrieval.py
├── utils/           # Fonctions utilitaires partagées
│   └── embeddings.py
├── config/          # Configuration globale
└── tests/           # Tests unitaires
```

### Choix d'architecture

#### Pattern Wrapper avec classe de base abstraite

Chaque modèle de topics (BERTopic, Top2Vec) est encapsulé dans un **wrapper** héritant de la classe de base `TopicModelEvaluator`. Ce pattern a été choisi pour :

- **Standardiser l'interface** : l'unification des méthodes (`getTopicWords()`, `getTopicsKeys()`, etc.) permet aux métriques de fonctionner de manière indépendante du modèle utilisé.
- **Faciliter l'extensibilité** : ajouter un nouveau modèle revient à créer un nouveau wrapper implémentant `TopicModelEvaluator`, sans modifier les métriques existantes.
- **Isoler la complexité** : chaque wrapper gère les spécificités de son modèle (API différente de BERTopic vs Top2Vec) de manière transparente pour le reste du code.

#### Métriques implémentées comme fonctions pures

Les trois métriques (`coherence`, `diversity`, `retrieval`) sont implémentées comme des **fonctions indépendantes** (et non des classes) prenant un `TopicModelEvaluator` en paramètre. Ce choix offre :

- **Simplicité** : une fonction sans état est plus facile à tester et à comprendre.
- **Composabilité** : les métriques peuvent être appelées indépendamment ou combinées librement.
- **Paramétrage explicite** : la fonction `diversity()` accepte par exemple une fonction de distance injectable (`cosine_similarity` par défaut), rendant le comportement facilement configurable sans sous-classer.

#### Séparation utils / metrics

Les calculs d'embeddings partagés (e.g. `calculCentroide`) sont isolés dans `utils/embeddings.py` pour éviter la duplication entre les métriques et conserver des fonctions de calcul réutilisables.

## Datasets

### 20 Newsgroups

Le dataset 20 Newsgroups est une collection de 18 846 articles de newsgroups répartis en 20 catégories thématiques (politique, religion, sport, science, informatique, etc.). Il constitue un benchmark classique pour la classification et le clustering de texte.

### AG News

Le dataset AG News regroupe 127 600 articles de presse (120 000 train + 7 600 test) issus de plus de 2 000 sources, répartis en 4 catégories : *World*, *Sports*, *Business* et *Sci/Tech*. Il est fréquemment utilisé comme benchmark de classification de texte court. Nous utilisons ici la version de train du dataset.

### ArXiv

Le dataset ArXiv est un ensemble de données de 31 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### Big Patent

Le dataset Big Patent Clustering (variante Jina AI) est un sous-ensemble du dataset Big Patent, contenant des brevets répartis en 9 catégories CPC (Cooperative Patent Classification). Il est conçu pour le clustering de documents longs. Nous utilisons ici la version "Big Patent Clustering" du dataset contenant 3 documents.

### BioRxiv

Le dataset BioRxiv (variante MTEB `biorxiv-clustering-p2p`) contient 53 787 paires titre+résumé d'articles de biologie préprint, répartis en 26 catégories disciplinaires (neurosciences, microbiologie, génomique, bioinformatique, etc.). Nous utilisons ici la variante "biorxiv-clustering-p2p" du dataset.

### ClusTREC-Covid

Le dataset ClusTREC-Covid est une adaptation du dataset TREC-COVID pour le clustering. Il contient 6 852 articles scientifiques sur la COVID-19 (titres + résumés), regroupés en 50 topics thématiques (e.g., *réponse du coronavirus aux changements météorologiques*). Nous utilisons ici la version "ClusTREC-Covid" du dataset.

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

### Observations Cohérence

D'après le graphique ci-dessous, on peut observer que BERTopic est plus cohérent que Top2Vec sur l'ensemble des datasets.

![Cohérence](images/coherence.png)

### Observations Retrieval

D'après le graphique ci-dessous, on peut observer que BERTopic est plus performant que Top2Vec sur l'ensemble des datasets.

![Retrieval](images/retrieval.png)

### Observations Diversité

La diversité mesure les distinctions entre les topics générés (absence de chevauchement dans leurs mots-clés). Les valeurs observées se situent entre **~0.899 et ~0.927**, indiquant une diversité globalement très élevée.

Plusieurs tendances se dégagent en fonction du nombre de topics :

- **k=10–25** : légère hausse avec un pic local (~0.922), le modèle génère peu de topics bien distincts.
- **k=25–60** : **forte chute jusqu'au minimum (~0.899 à k=60)** — zone critique où le modèle crée des topics redondants partageant des mots-clés similaires.
- **k=60–125** : remontée progressive avec des oscillations, le modèle retrouve de la spécialisation.
- **k=150–200** : **tendance haussière nette** vers le maximum (~0.927), les topics deviennent de plus en plus spécialisés et distincts.

> **Conclusion** : la diversité seule favorise les grands `k`, mais doit être croisée avec la cohérence (qui tend à baisser quand `k` est trop élevé) pour identifier le meilleur compromis.

On remarque que si le nombre de topics est faible, la diversité est élevée car les topics sont plus généraux.
Si le nombre de topics est élevé, la diversité est aussi élevée car les topics sont plus spécifiques.

![Diversité](images/diversite.png)
