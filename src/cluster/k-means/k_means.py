import numpy as np
import pandas as pd
import base
from enum import Enum
from math import *

class DistanceMetric(enum):
    eucledian = lambda p1, p2 = sqrt(sum((x1-x2)**2 for x1, x2 in zip(p1, p2)))

class KMeans(Clusterer):

    def __init__(self, k, dist_func=lambda , eps=1e-4, doGraph=False):
        """ Initialize K-Means Clusterer
        k - number of clusters to fit
        dist_func - the distance function
        eps - stopping criterion tolerance
        doGraph - graphs after fitting
        """
        self.k = k
        self.dist_func = dist_func
        self.eps = eps
        self.doGraph = doGraph

    def fit(self, data):
        """ Run the k-means algorithm
        data - an NxD pandas DataFrame


        returns: a tuple containing
            mu - a KxD ndarray containing the learned means
            cluster_assignments - an N-vector of each point's cluster index
            cost - the total cost of all the points to their assigned cluster
        """
        n, d = data.shape
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
        assigns = np.zeros(data.shape[0])
        prev_cost = -1
        cost = 0
        while abs(cost-prev_cost) > eps:
            prev_cost = cost
            # Assigns every point to the nearest centroid
            for i in range(len(assigns)):
                assigns[i] = np.argmin(np.sum(np.square(np.tile(data[i],(k,1)) - mu),axis=1))
            cost = 0
            # For every cluster calculates the distance to data points and add to the total cost
            for j in range(k):
                mu[j] = np.dot((assigns == j).astype(int),data) / sum((assigns == j).astype(int))
                cost += np.dot((assigns == j).astype(int),np.sum(np.square(data-np.tile(mu[j],(len(data),1))),axis=1))

        return (mu, assigns, cost)
