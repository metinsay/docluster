import numpy as np
import pandas as pd
from utils.constants.distance_metric import DistanceMetric
from math import *
from clusterer import Clusterer
import random

class KMeans(Clusterer):

    def __init__(self, k, dist_func=DistanceMetric.eucledian, eps=1e-4, do_graph=False):
        """ Initialize K-Means Clusterer
        k - number of clusters to fit
        dist_func - the distance function
        eps - stopping criterion tolerance
        do_graph - graphs after fitting
        """
        self.k = k
        self.dist_func = dist_func
        self.eps = eps
        self.do_graph = do_graph
        self.clusters = None
        self.centroids = None
        self.cost = None

    def fit(self, data):
        """ Run the k-means algorithm
        data - an NxD pandas DataFrame

        returns: a tuple containing
            centroids - a KxD ndarray containing the learned means
            clusters - an N-vector of each point's cluster index
            cost - the total cost of all the points to their assigned cluster
        """
        n, d = data.shape
        data = np.array(list(map(lambda item: np.array(item.todense()),data)))
        # randomly choose k points as initial centroids
        centroids = data[random.sample(range(data.shape[0]), self.k)]
        clusters = np.zeros(n)
        prev_cost = -1
        cost = 0
        while abs(cost-prev_cost) > self.eps:
            prev_cost = cost
            # Assigns every point to the nearest centroid
            for i in range(n):
                tiled_data_point = np.tile(data[i], (self.k, 1))
                clusters[i] = np.argmin(self.dist_func(tiled_data_point, centroids))
            cost = 0
            # For every cluster calculates the distance to data points and add to the total cost
            for j in range(self.k):
                data_with_j_assignment = (clusters == j).astype(int)
                centroids[j] = np.dot(data_with_j_assignment, data) / sum(data_with_j_assignment)

                tiled_centroid = np.tile(centroids[j], (n, 1))
                cost += np.dot(data_with_j_assignment, self.dist_func(data, tiled_centroid))

        self.centroids, self.clusters, self.cost = centroids, clusters, cost

        if self.do_graph:
            Grapher().plot_voronoi(data, clusters=clusters, centroids=centroids, title="K-means clustering with k= " + str(self.k))

        return (centroids, clusters, cost)
