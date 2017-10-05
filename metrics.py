import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.metrics import ndcg_score

'''
 These are for rating predictions.
'''

# Mean Absolute Error - sklearn.metrics.mean_absolute_error
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error

# Root Mean Square Error
def root_mean_square_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

'''
 These are for top-n recommendations.
'''

# Precision at k
def precision(y_true, y_pred, k):
    if len(y_pred)>k:
        y_pred = y_pred[:k]

    if not y_true:
        return 0.0

    return len(set(y_pred).intersection(y_true)) /  float(len(y_pred))

def precision_k(y_true, y_pred, k):
    """
    P@k between two lists of lists of items.
    
    y_true : list
        A list of lists of elements that are to be predicted 
    y_pred : list
        A list of lists of predicted elements
    k : int 
        The maximum number of predicted elements
    """
    set(b1).intersection(b2)
    return np.mean([precision(a, p, k)  for a,p in zip(y_true, y_pred)])


# Mean Average Precision at k

# First, Average Precision at k
def average_precision(y_true, y_pred, k):
    """
    AP@k between two lists of items.

    y_true : list
        A list of elements that are to be predicted (order doesn't matter)
    y_pred : list
        A list of predicted elements (order does matter)
    k : int
        The maximum number of predicted elements
    """
    if len(y_pred)>k:
        y_pred = y_pred[:k]

    if not y_true:
        return 0.0

    score = 0.0
    tp = 0.0
    for i,p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            tp += 1.0
            score += tp / (i+1.0)

    return score / min(len(y_true), k)

def mean_average_precision(y_true, y_pred, k):
    """
    MAP@k between two lists of lists of items.
    
    y_true : list
        A list of lists of elements that are to be predicted 
    y_pred : list
        A list of lists of predicted elements
    k : int 
        The maximum number of predicted elements
    """
    return np.mean([average_precision(a,p,k) for a,p in zip(y_true, y_pred)])

# Truncated Normalized Discounted Cumulative Gain (NDCG@K)

# First, we need DCG@K
def dcg_k(y_true, y_pred, k):
    """
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_pred : array-like, shape = [k]
        y_pred[i] is the ith top-scored document.
    k : int
        Rank.
    """
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rel = y_true[y_pred]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(y_pred)) + 2)
    return np.sum(gains / discounts)


def ndcg_k(y_true, y_pred, k):
    """Normalized discounted cumulative gain (NDCG) at rank k

    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_pred : array-like, shape = [k]
        y_pred[i] is the ith top-scored document.
    k : int
        Rank.
    """
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    best_y_pred = np.argsort(y_true)[::-1]
    best = dcg_k(y_true, best_y_pred[:k], k)
    return dcg_k(y_true, y_pred, k) / best




