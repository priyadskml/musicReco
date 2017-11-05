import numpy as np
from scipy.sparse import csr_matrix

from mrec.sparse import fast_sparse_matrix
from mrec.mf.recommender import MatrixFactorizationRecommender

class MSWRMF(MatrixFactorizationRecommender):
    """
    Parameters
    ==========
    d : int
        Number of latent factors.
    alpha : float
        Confidence weight, confidence c = 1 + alpha*r where r is the observed "rating".
    lbda : float
        Regularization constant.
    num_iters : int
        Number of iterations of alternating least squares.
    """

    def __init__(self,d,k,a, alpha=1,lbda=0.015,num_iters=15):
        self.d = d
        self.k = k
        self.a = a
        self.alpha = alpha
        self.lbda = lbda
        self.num_iters = num_iters

        # R, Rte, S, D, k, alpha, l, eta, iterations

    def __str__(self):
        return 'WRMFRecommender (d={0},alpha={1},lambda={2},num_iters={3})'.format(self.d,self.alpha,self.lbda,self.num_iters)

    def init_factors(self,num_factors,assign_values=True):
        if assign_values:
            return self.d**-0.5*np.random.random_sample((num_factors,self.d))
        return np.empty((num_factors,self.d))

    def fit(self,train,S,item_features=None):
        """
        Learn factors from training set. User and item factors are
        fitted alternately.
        Parameters
        ==========
        train : scipy.sparse.csr_matrix or mrec.sparse.fast_sparse_matrix
            User-item matrix.
        S: item similaities
        item_features : array_like, shape = [num_items, num_features]
            Features for each item in the dataset, ignored here.
        """
        if type(train) == csr_matrix:
            train = fast_sparse_matrix(train)

        num_users,num_items = train.shape

        self.U = self.init_factors(num_users,False)  # don't need values, will compute them
        self.V = self.init_factors(num_items) # Items
        for it in xrange(self.num_iters):
            print 'iteration',it
            # fit user factors
            VV = self.V.T.dot(self.V)
            for u in xrange(num_users):
                # get (positive i.e. non-zero scored) items for user
                indices = train.X[u].nonzero()[1]
                if indices.size:
                    self.U[u,:] = self.update(indices,self.V,VV)
                else:
                    self.U[u,:] = np.zeros(self.d)
            # fit item factors
            UU = self.U.T.dot(self.U)
            for i in xrange(num_items):
                indices = train.fast_get_col(i).nonzero()[0]
                if indices.size:
                    self.V[i,:] = self.update(indices,self.U,UU)
                else:
                    self.V[i,:] = np.zeros(self.d)

    def update(self,indices,H,HH):
        """
        Update latent factors for a single user or item.
        """
        Hix = H[indices,:]
        M = HH + self.alpha*Hix.T.dot(Hix) + np.diag(self.lbda*np.ones(self.d))
        return np.dot(np.linalg.inv(M),(1+self.alpha)*Hix.sum(axis=0))

