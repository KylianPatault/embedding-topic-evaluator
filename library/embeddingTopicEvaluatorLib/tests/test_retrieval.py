# Tests pour la métrique de retrieval

import numpy as np
import json
from ..metrics.retrieval import calculate_retrieval_metrics

TARGET = np.array([0, 1, 2, 3, 4])
PROBS = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.5, 0.4, 0.3, 0.2, 0.1],
                  [0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.6, 0.5, 0.4, 0.3, 0.2]])

def test_retrieval_metrics():
    metrics = calculate_retrieval_metrics(TARGET, PROBS)
    assert isinstance(metrics, dict)
    print(json.dumps(metrics, indent=4))

if __name__ == '__main__':
    test_retrieval_metrics()
