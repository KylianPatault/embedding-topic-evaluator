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
Durant nos entraînements, nous avons fait varier le paramètre **nr_topics** qui correspond au nombre de topics que le modèle doit créer. 

### Top2Vec

Top2Vec est un algorithme de modélisation thématique et de recherche sémantique. Il détecte automatiquement les thèmes présents dans le texte et génère des vecteurs de thèmes, de documents et de mots représentés conjointement dans un même espace vectoriel.

Pour entraîner ce modèle nous avons choisie les paramètres suivants (disponible dans le fichier [config](./library/embeddingTopicEvaluatorLib/config/config.py):
```
"UMAP" : {
    "n_neighbors" : 15, # Taille du voisinage local (plus élevé = vision plus globale)
    "n_components" : 10, # Dimension de sortie pour le clustering
    "min_dist" : 0.0, # Distance min entre les points
    "metric" : "cosine" # Métrique de distance
},
        
"EmbeddingModel" : EMBEDDING_MODEL,
        
"TOP2VEC" : {
    # Modèle de Sentence Transformers utilisé pour les embeddings
    "embedding_model" : "paraphrase-multilingual-MiniLM-L12-v2",
    "min_count" : 5, # Nombre minimum d'occurrences d'un mot pour être pris en compte
    "nr_topics" : ..., # Nombre de topics à générer
    "verbose" : True, # Affiche la barre de progression
    "n_components" : 10 # Nombre de mots par Topics
}
```
Pour ce modèle, nous avons choisi d'utiliser le model [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) car c’était l’option la plus performante à ce moment-là. 

### BERTopic

BERTopic est un framework moderne de modélisation thématique qui répond à de nombreuses limites des approches traditionnelles. Développé par Maarten Grootendorst , il utilise des plongements (embeddings) basés sur des transformeurs (comme BERT) pour comprendre la signification sémantique des documents et les regrouper en clusters en fonction de leur contexte, plutôt que de se baser uniquement sur la fréquence des mots.

Pour entraîner ce modèle nous avons choisie les paramètres suivants (disponible dans le fichier [config](./library/embeddingTopicEvaluatorLib/config/config.py):
```
"UMAP" : {
    "n_neighbors" : 50, # Taille du voisinage local (plus élevé = vision plus globale)
    "n_components" : 10, # Dimension de sortie pour le clustering
    "min_dist" : 0.0, # Distance min entre les points
    "metric" : "cosine" # Métrique de distance
},
"HDBSCAN" : {
    "min_cluster_size" : 50, # Taille min d'un topic
    "min_samples" : 1, # Sensibilité au bruit (plus bas = moins d'exclus)
    "metric" :"euclidean", # Distance standard après UMAP
    "cluster_selection_method" : "eom", # Sélection des clusters les plus denses
    "prediction_data" : True # Permet de classer de nouveaux documents
},
"EmbeddingModel" : EMBEDDING_MODEL, 
    
"BERTopic" : {
    # Modèle de Sentence Transformers utilisé pour les embeddings
    "embedding_model" : "all-mpnet-base-v2",
    "nr_topics" : ... , # Réduction automatique du nombre de topics si nécessaire
    "verbose" : True # Affiche la barre de progression
}
```
Pour ce modèle, nous avons choisi d'utiliser le model [all-mpnet-base-v2](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) car c’était l’option la plus performante à ce moment-là.

## Métriques d'évaluation

Ce projet utilise quatre métriques d'évaluation pour évaluer les performances des modèles. Ces métriques sont la cohérence, le retrieval, la diversité et la cohésion.
Pour calculer ces métriques, on utilise un modèle différant, cela permet de comparer les 2 modèles indépendamment de leurs représentations internes des embeddings. Bien sûr, il est toujours possible de calculer les métriques en fonction du modèle de production d'embedding interne. Pour information, la métrique de cohésion est calculée sur les embeddings interne. 

Le modèle choisi pour faire la comparaison est [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), nous avons choisi ce modèle, car il a été entraîné sur un très grand nombre de données de type texte plus de 1 milliard. Ce modèle a été créé pour encoder des phrases et de courts paragraphes. À partir d'un texte d'entrée, il produit un vecteur qui capture l'information sémantique. Ce vecteur peut être utilisé pour la recherche d'informations, le regroupement de données ou l'analyse de similarité entre phrases.

### Cohérence

Cette métrique mesure la cohérence sémantique de chaque topic en calculant la similarité cosinus entre les embeddings de toutes les paires uniques de mots qui le composent (triangle supérieur de la matrice de similarité, diagonale exclue), puis en faisant la moyenne de ces similarités. Un score proche de 1 indique un topic cohérent dont les mots sont fortement liés sémantiquement.
Par exemple : 
Le topic : ["voiture", "camion", "véhicule", "roue"] doit avoir un score de cohérence élevé car tous les mots sont liés sémantiquement.
Alors que, le topic : ["voiture", "pomme", "véhicule", "roue"] doit avoir un score de cohérence faible car les mots ne sont pas tous liés sémantiquement.

### Retrieval

Cette métrique calcule la fidélité aux documents du topic en utilisant la méthode TREC. Cette approche soumet des requêtes dont chacune d'elles nous retourne les documents correspondant au topic fourni. On utilise deux métriques pour évaluer le retrieval :

- MAP (Mean Average Precision) : C'est la moyenne des précisions moyennes pour chaque requête.
- NDCG (Normalized Discounted Cumulative Gain) : C'est la moyenne des gains cumulés normalisés pour chaque requête.

### Diversité

Cette métrique permet de vérifier si le modèle créé des topics éloigner du point de vue des embeddings. On fait cela, pour vérifier si notre modelé réparti bien les topics dans l'espace, car si deux topics sont trop rapprocher, il pourrait y avoir une confusion lors du placement d'un nouveau document. Pour calculer cette métrique, il y a 2 cas le premier, on utilise le modèle d'embedding interne qui vas calcul le centroide des topics (la moyenne des embeddings des mots qui définisse le topics), puis on va comparer ces centroide entre eu avec une similarité cosinus. On fini par prendre la valeur de similarité du centroide le plus proche qu'on va soustraire à 1, puis on fait la moyenne pour chaque centroide. Le deuxième cas est très semblable au premier sauf qu'on va transformer les mots du topic une phrase par exemple le topic : ["voiture", "camion", "véhicule", "roue"] va devenir "voiture camion véhicule roue", on va créer un embedding de cette phrase avec un modèle externe puis le reste est identique à la première méthode. Le but de cette métrique est donc d'être maximisé. 

### Cohésion

Cette métrique compare pour un topic si en calculant le centroide des mots qui le définisse (la moyenne des embeddings des mots qui définisse le topics) et l'embedding de la phrase composer des mots qui définissent le topics. Si on se retrouve avec une similarité cosinus élever, cela veux dire les mots qui définissent notre topics le représente bien. Par défaut, cette métrique est calculée sur les embedding produit par le modèle interne. 

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
│   ├── cohesion.py
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

Les quatres métriques (`coherence`, `diversity`, `retrieval`, `cohesion`) sont implémentées comme des **fonctions indépendantes** (et non des classes) prenant un `TopicModelEvaluator` en paramètre. Ce choix offre :

- **Simplicité** : une fonction sans état est plus facile à tester et à comprendre.
- **Composabilité** : les métriques peuvent être appelées indépendamment ou combinées librement.
- **Paramétrage explicite** : la fonction `diversity()` accepte par exemple une fonction de distance injectable (`cosine_similarity` par défaut), rendant le comportement facilement configurable sans sous-classer.

#### Séparation utils / metrics

Les calculs d'embeddings partagés (e.g. `calculCentroide`) sont isolés dans `utils/embeddings.py` pour éviter la duplication entre les métriques et conserver des fonctions de calcul réutilisables.

## Datasets

### [20 Newsgroups](https://www.kaggle.com/datasets/crawford/20-newsgroups/data)

Le dataset 20 Newsgroups est une collection de 18 846 articles de newsgroups répartis en 20 catégories thématiques (politique, religion, sport, science, informatique, etc.). Il constitue un benchmark classique pour la classification et le clustering de texte.

### [AG News](https://huggingface.co/datasets/wangrongsheng/ag_news)

Le dataset AG News regroupe 127 600 articles de presse (120 000 train + 7 600 test) issus de plus de 2 000 sources, répartis en 4 catégories : *World*, *Sports*, *Business* et *Sci/Tech*. Il est fréquemment utilisé comme benchmark de classification de texte court. Nous utilisons ici la version de train du dataset.

### [ArXiv](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s)

Le dataset ArXiv est un ensemble de données de 31 articles de presse répartis en 4 catégories. Il est utilisé pour évaluer les performances des modèles de classification de texte.

### [Big Patent](https://huggingface.co/datasets/jinaai/big-patent-clustering)

Le dataset Big Patent Clustering (variante Jina AI) est un sous-ensemble du dataset Big Patent, contenant des brevets répartis en 9 catégories CPC (Cooperative Patent Classification). Il est conçu pour le clustering de documents longs. Nous utilisons ici la version "Big Patent Clustering" du dataset contenant 3 documents.

### [BioRxiv](https://huggingface.co/datasets/mteb/biorxiv-clustering-p2p)

Le dataset BioRxiv (variante MTEB `biorxiv-clustering-p2p`) contient 53 787 paires titre+résumé d'articles de biologie préprint, répartis en 26 catégories disciplinaires (neurosciences, microbiologie, génomique, bioinformatique, etc.). Nous utilisons ici la variante "biorxiv-clustering-p2p" du dataset.

### [ClusTREC-Covid](https://huggingface.co/datasets/Uri-ka/ClusTREC-Covid)

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
