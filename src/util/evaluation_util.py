from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def prf(real: np.ndarray, pred: np.ndarray):
    p = precision_score(real, pred)
    r = recall_score(real, pred)
    f = f1_score(real, pred)

    return p, r, f