import numpy as np
import pandas as pd
from utils import DistanceMetric, Grapher
from math import *
from .clusterer import Clusterer
import random

class KMeans(Clusterer):

    def __init__(self, k, dist_metric=DistanceMetric.eucledian, eps=1e-4, do_plot=False):
        """ Initialize K-Means Clusterer
        k - number of clusters to fit
        dist_metric - the distance metric
        eps - stopping criterion tolerance
        do_plot - voronoi plot after fitting
        """
        self.k = k
        self.dist_metric = dist_metric
        self.eps = eps
        self.do_plot = do_plot
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
                clusters[i] = np.argmin(self.dist_metric(tiled_data_point, centroids))
            cost = 0

            # For every cluster calculates the distance to data points and add to the total cost
            for j in range(self.k):
                centroids[j] = np.mean(data[clusters == j], axis=0)
            cost = np.sum(np.square(data - centroids[clusters.astype('int64')]))

        self.centroids, self.clusters, self.cost = centroids, clusters, cost

        if self.do_plot:
            Grapher().plot_voronoi(data, n_clusters=self.k, clusters=clusters, centroids=centroids, title="K-means clustering with k= " + str(self.k))

        return (centroids, clusters, cost)



    def get_distances_btw_centroids(self, dist_metric=None, do_plot=False):

        if not self.cost:
            assert('You need to fit the data first in order to get distances between centroids.')
        else:
            # Check if distance metric is changed
            dist_metric = dist_metric if dist_metric else self.dist_metric
            dists = np.zeros((self.k, self.k))

            # Subtract a selected cetroid from each centroid
            for i, centroid in enumerate(self.centroids):
                tiled_centroid = np.tile(centroid, (self.k, 1))
                dists[i] = dist_metric(tiled_centroid, self.centroids)

            if do_plot:
                Grapher().plot_heat_map(list(dists), "Centroids' Distance Heat Map")

            return dists
