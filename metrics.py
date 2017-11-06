import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
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

#     if not y_true:
#         return 0.0
#     y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    return len(set(y_pred).intersection(set(y_true)))/  float(len(y_pred))

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
#     set(b1).intersection(b2)
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

    # if y_true.all() is None or len(y_true) < 1:
    #     return 0.0

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
def find_dcg(element_list, k=10):
    """
    Discounted Cumulative Gain (DCG)
    The definition of DCG can be found in this paper:
        Azzah Al-Maskari, Mark Sanderson, and Paul Clough. 2007.
        "The relationship between IR effectiveness measures and user satisfaction."
    Parameters:
        element_list - a list of ranks Ex: [5,4,2,2,1]
    Returns:
        score
    """
    score = 0.0
#     print element_list
    for order, rank in enumerate(element_list):
        if order < k:
            score += (float(rank))/math.log((order+2), 2)
    return score


def find_ndcg(reference, hypothesis, k=10):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG(hypothesis)/DCG(reference)
    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        ndcg_score  - normalized score
    """
#     print hypothesis
#     print reference
#     res = find_dcg(hypothesis)/find_dcg(reference)
#     print res
    return find_dcg(hypothesis)/find_dcg(reference)

def ndcg_k(y_true, y_pred, test_r, k=10):
    ndcgs = []
    # For ground truth y_rec 
    # For the actual
    for i in range(len(y_pred)):
        ndcgs.append(find_ndcg(test_r[i, y_true[i].flatten()], test_r[i,y_pred[i]], k=10))
    return np.mean(ndcgs)

