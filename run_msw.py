D = 128
k = 5
alpha = 0.5
l = 1
num_iters = 50

print k


from utils import * 
import numpy as np
import pandas as pd
from metrics import * 

URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
f_name = "lastfm-dataset-1K.tar.gz"
dir_name = "lastfm-dataset-1K"
dataset_f_name = "userid-timestamp-artid-artname-traid-traname.tsv"

from scipy.sparse import csr_matrix
from scipy import io


# Now we load the ratings
implicit_ratings = io.mmread("ratings.mtx")

print("Loaded ratings...")

# Test-train split 
TRAIN_SIZE = 0.80
# Create boolean mask
# np.random creates a vector of random values between 0 and 1
# Those values are filtered to create a binary mask
msk = np.random.rand(implicit_ratings.shape[0],implicit_ratings.shape[1]) < TRAIN_SIZE
r = np.zeros(implicit_ratings.shape)
print (msk.shape)
 
train_ratings = implicit_ratings.copy()
test_ratings = implicit_ratings.copy()
    
train_ratings[~msk] = r[~msk]
test_ratings[msk] = r[msk] # inverse of boolean mask

print("Split ratings...")

# We take the test ratings THIS IS FOR RECOMMENDATION TASK
y_rec = []
# now we need to sort each row
for u in test_ratings:
    indices = np.nonzero(u)[0]
    nz = indices[np.argsort(u[indices], axis=None)]
    y_rec.append(nz[::-1])
print len(y_rec)
print y_rec[:1]

ratings = np.array(implicit_ratings)
test_r = np.array(test_ratings)
train_r = np.array(train_ratings) 

words, vec = load_vectors('song2vec.txt')

with open("trackid-index.csv") as f:
    ordered = f.read().splitlines() 

ordered_len = len(ordered)
words_len = len(words) - 2

print vec[0].shape
dim =100
print dim
vecs = []
j = 0
for tid in ordered:
    if tid in words:
        i = words.index(tid)
        vecs.append(vec[i])
    else:
        vecs.append(np.random.randn((dim)))
    j += 1
    if j % 10000 == 0:
        print j
ordered_vecs = np.array(vecs)
print ordered_vecs.shape
# ordered_vecs = np.random.normal(scale=1./8, size=(train_ratings.shape[1], 10))
# This will give an ordered version of the vectors 
print("Ordered vectors...")

print("Now we train...")

from mswrmf import *
mswrmf = MSWRMF(d=D,k=k, a=alpha, num_iters=num_iters)
# R, Rte, S, D, k, alpha, l, eta, iterations)
from scipy.sparse import csr_matrix 
training_process = mswrmf.fit(csr_matrix(train_r), ordered_vecs)

n_rec = mswrmf.batch_recommend_items(csr_matrix(train_r),max_items=10,return_scores=False)
# n_rec = np.array(n_rec)
# n_rec[msk] = r[msk]

print ("MAP@k",mean_average_precision(y_rec, n_rec, 10))
print ("P@k",precision_k(y_rec, n_rec, 10))
# NDCG also takes the ratings into account  test_ratings
print ("NDCG@k",ndcg_k(y_rec, n_rec, test_r, 10))




