from bertopic import BERTopic

def getWordsTopics(topicsKey :int, model : BERTopic) -> list :
    return [word for word,_ in model.get_topic(topicsKey)]