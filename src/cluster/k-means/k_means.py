import numpy as np
import pandas as pd

def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an NxD pandas DataFrame
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
        assigns = np.zeros(data.shape[0])
        prev_cost = -1
        cost = 0
        while abs(cost-prev_cost) > 0.00001:
            prev_cost = cost
            for i in range(len(assigns)):
                assigns[i] = np.argmin(np.sum(np.square(np.tile(data[i],(k,1)) - mu),axis=1))
            cost = 0
            for j in range(k):
                mu[j] = np.dot((assigns == j).astype(int),data) / sum((assigns == j).astype(int))
                cost += np.dot((assigns == j).astype(int),np.sum(np.square(data-np.tile(mu[j],(len(data),1))),axis=1))

        return (mu,assigns)
