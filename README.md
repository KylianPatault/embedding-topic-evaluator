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
- [Word Intrusion Test](#word-intrusion-test)
  - [Choix des mots](#choix-des-mots)
  - [Choix de l'intrus](#choix-de-lintrus)

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

Cette métrique calcule la fidélité aux documents du topic en utilisant la méthode TREC. Cette approche soumet des requêtes dont chacune d'elles nous retourne les documents correspondant au topic fourni.

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
