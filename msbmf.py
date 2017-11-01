class MSBMF():
    
    def __init__(self, R, Rte, S, D, k, alpha, l, eta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        
        Arguments
        - R (ndarray)     : user-item rating matrix
        - Rte (ndarray)   : user-item rating matrix for testing
        - S (ndarray)     : song similarity matrix
        - D (int)         : number of latent dimensions
        - k (int)         : k nearest neighbors to consider for any song i 
        - l (float)       : regularization parameter
        - eta (float)     : learning rate
        - alpha (float)   : weight on song similarities
        - iterations (int): No. of iterations to train over
        """
        
        self.R = R
        self.Rte = Rte
        self.S = S
        self.num_users, self.num_items = R.shape
        self.D = D
        self.k = k
        self.l = l
        self.eta = eta
        self.alpha = alpha
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.D, size=(self.num_users, self.D))
        self.Q = np.random.normal(scale=1./self.D, size=(self.num_items, self.D))
        
        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Similairities are pretrained; is a num_items x num_items matrix
        
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))
        
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.Rte.nonzero()
        predicted = self.full_matrix()
        error = 0
        count=0
        for x, y in zip(xs, ys):
            err =  pow(self.Rte[x, y] - predicted[x, y], 2)
            if(err>0):
                count += 1
                error += err               
        error = error/count
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            
            # Update biases
            self.b_u[i] += self.eta * (e - self.l * self.b_u[i])
            self.b_i[j] += self.eta * (e - self.l * self.b_i[j])
            
            # Update user and item latent feature matrices
            self.P[i, :] += self.eta * (e * self.Q[j, :] - self.l * self.P[i,:])
            self.Q[j, :] += self.eta * (e * self.P[i, :] - self.l * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)