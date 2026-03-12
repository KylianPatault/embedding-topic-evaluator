import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import json

from embeddingTopicEvaluatorLib.metrics.diversity import diversity
from embeddingTopicEvaluatorLib.metrics.coherence import coherence
from embeddingTopicEvaluatorLib.metrics.retrieval import retrieval
from embeddingTopicEvaluatorLib.metrics.cohesion import cohesion
from embeddingTopicEvaluatorLib.config.config import settings

from embeddingTopicEvaluatorLib.models.top2vec_wrapper import TopicModelEvaluatorTop2Vec
from embeddingTopicEvaluatorLib.models.bertopic_wrapper import TopicModelEvaluatorBERTopic

from datasets import load_dataset

def generate_top2vec_models(param, base_config, docs):
    """
    Génère une liste de modèles d'évaluation Top2Vec en faisant varier 
    le paramètre voulu.
    """
    topic_models = []
    
    # On itère sur chaque paramètre défini dans la liste
    for param in param:
        
        # Copie profonde pour éviter de modifier la configuration des modèles précédents
        current_config = copy.deepcopy(base_config)
        
        # Mise à jour du nombre de topics dans la configuration courante
        current_config["TOP2VEC"]["nr_topics"] = int(param)
        
        # Création de l'évaluateur avec la configuration spécifique et ajout à la liste
        model_evaluator = TopicModelEvaluatorTop2Vec(docs, current_config)
        topic_models.append(model_evaluator)
        
    return topic_models

def generate_bertopic_models(param, base_config, docs):
    """
    Génère une liste de modèles d'évaluation BERTopic en faisant varier 
    le paramètre voulu.
    """
    topic_models = []
    
    # On itère sur chaque paramètre défini dans la liste
    for param in param:
        
        # Copie profonde pour éviter de modifier la configuration des modèles précédents
        current_config = copy.deepcopy(base_config)
        
        # Mise à jour du nombre de topics dans la configuration courante
        current_config["BERTopic"]["nr_topics"] = int(param)
        
        # Création de l'évaluateur avec la configuration spécifique et ajout à la liste
        model_evaluator = TopicModelEvaluatorBERTopic(current_config)
        topic_models.append(model_evaluator)
        
    return topic_models

def evaluate_all_models(topic_models, docs):
    """
    Évalue une liste de modèles sur un corpus de documents.
    """
    topics_hdbscans = []
    probs_hdbscans = []

    # On parcourt chaque modèle généré précédemment
    for model in topic_models:
        
        # On lance l'évaluation du modèle sur le corpus
        # La méthode evaluate retourne les topics et leurs probabilités
        topics_hdbscan, probs_hdbscan = model.evaluate(docs)
        
        # Ajout des résultats récupérés dans nos listes
        topics_hdbscans.append(topics_hdbscan)
        probs_hdbscans.append(probs_hdbscan)
        
    return topics_hdbscans, probs_hdbscans

def compute_model_metrics(topic_models, docs):
    """
    Calcule les métriques de cohérence, de recherche (retrieval) et de diversité 
    pour une liste de modèles de topics.
    """
    coherences = []
    retrievals = []
    diversities = []
    cohesions = []

    # On évalue les métriques pour chaque modèle
    for model in topic_models:
        
        # Calcul de la Cohérence
        coherence_stack = coherence(model)
        # On exclut le topic -1 (qui représente le bruit/outliers dans BERTopic)
        valeurs_sans_outliers = [valeur for cle, valeur in coherence_stack.items() if cle != -1]
        # On calcule la moyenne des cohérences des topics "valides"
        coherence_mean = np.mean(valeurs_sans_outliers)
        coherences.append(coherence_mean)

        # Calcul du Retrieval
        # Nécessite le modèle et les documents pour évaluer la pertinence
        retrieval_stack = retrieval(model, docs)
        retrievals.append(retrieval_stack)
        
        # Calcul de la Diversité
        diversity_stack = diversity(model)
        diversities.append(diversity_stack)

        # Calcul de la cohésion
        cohesion_stack = cohesion(model)
        # On calcule la moyenne des cohésions des topics
        cohesion_mean = np.mean(cohesion_stack)
        cohesions.append(cohesion_mean)
        

    return coherences, retrievals, diversities, cohesions

def graph(param, metric, title, xlabel, ylabel, label, save_name):
    fig = plt.figure(figsize=(10, 6))

    plt.plot(param, metric, marker='o', linestyle='-', color='b', label=label)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.legend()
    plt.show()
    fig.savefig(save_name)

def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    db = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    docs = db[config["dataset"]["db"]]

    nr_topics_param = np.linspace(config["param"]["nr_topic_min"], 
                                  config["param"]["nr_topic_max"], 
                                  config["param"]["nombre_nr_topic"], 
                                  dtype=int)
    
    if (config["model"]["name"] == "top2vec"):
        docs = list(docs)
    
        config_model = settings.TOP2VEC_CONFIG

        # Test en faisant varié le paramètre nombre de topics
        topic_models = generate_top2vec_models(nr_topics_param, config_model, docs)
        
        topics_hdbscans, probs_hdbscans = evaluate_all_models(topic_models, docs)

    elif (config["model"]["name"] == "BERTopic"):
        config = settings.BERTOPIC_CONFIG_HDBSCAN

        # Test en faisant varié le paramètre nombre de topics
        topic_models = generate_bertopic_models(nr_topics_param, config, docs)
        
        topics_hdbscans, probs_hdbscans = evaluate_all_models(topic_models, docs)

    else:
        raise "Model non pris en compte (seul top2vec et BERTopic le sont actuellement)."
        
    coherences, retrievals, diversities, cohesions = compute_model_metrics(topic_models, docs)
    
    title = 'Évolution de la cohérence en fonction du nombre de topics'
    xlabel = 'Nombre de topics'
    ylabel = 'Cohérence moyenne'
    label = 'Cohérence'
    save_name = config["stock_result"]["coherence"]
    graph(nr_topics_param, coherences, title, xlabel, ylabel, label, save_name)
    
    mean_maps = [np.mean([val['map'] for val in res.values()]) for res in retrievals]
    mean_ndcgs = [np.mean([val['ndcg'] for val in res.values()]) for res in retrievals]
    
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(nr_topics_param, mean_maps, marker='o', linestyle='-', color='g', label='MAP moyen')
    plt.plot(nr_topics_param, mean_ndcgs, marker='s', linestyle='-', color='r', label='NDCG moyen')
    
    plt.title("Évolution des performances de retrieval en fonction du nombre de topics", fontsize=14)
    plt.xlabel("Nombre de topics", fontsize=12)
    plt.ylabel("Score moyen", fontsize=12)
    
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    
    plt.legend()
    plt.show()
    
    fig.savefig(config["stock_result"]["retrieval"])
    
    title = 'Évolution de la diversité en fonction du nombre de topics'
    xlabel = 'Nombre de topics'
    ylabel = 'Diversité moyenne'
    label = 'Diversité'
    save_name = config["stock_result"]["diversite"]
    graph(nr_topics_param, diversities, title, xlabel, ylabel, label, save_name)
    
    title = 'Évolution de la cohésion en fonction du nombre de topics'
    xlabel = 'Nombre de topics'
    ylabel = 'Cohésion moyenne'
    label = 'Cohésion'
    save_name = config["stock_result"]["cohesion"]
    graph(nr_topics_param, cohesions, title, xlabel, ylabel, label, save_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic_run")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    main(parser.parse_args())