D = 128
k = 5
alpha = 0.5
l = 1



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

words, vec = load_vectors('song2vec.txt')

with open("trackid-index.csv") as f:
	ordered = f.read().splitlines() 

ordered_len = len(ordered)
words_len = len(words) - 2

# indices = []
# i = 0
# for tid in words:
# 	if tid in ordered:
# 		indices.append(ordered.index(tid)) 
# 	if i % 10000 == 0:
# 		print i
# 	i += 1

# ordered_vecs = vec[indices]

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

# This will give an ordered version of the vectors 
print("Ordered vectors...")


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
# y_rec = []
# # now we need to sort each row
# for u in test_ratings:
#     indices = np.nonzero(u)[1]
#     nz = indices[np.argsort(u[:,indices], axis=None)]
#     y_rec.append(nz[::-1])
# print len(y_rec)
# print y_rec[:1]

ratings = np.array(implicit_ratings)
test_r = np.array(test_ratings)
train_r = np.array(train_ratings) 

print("Now we train...")

from msbmf import *
msmf = MSBMF(train_r, test_r, S=ordered_vecs, D=D,k=k, alpha=alpha, l=l,eta=0.05, iterations=20)
# R, Rte, S, D, k, alpha, l, eta, iterations)
training_process = msmf.train()

predictions = msmf.full_matrix()
predictions[msk] = r[msk]

print "RMSE: %f" % root_mean_square_error(test_r, predictions)
print "MAE: %f" % mean_absolute_error(test_r, predictions)





