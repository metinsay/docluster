import random
from math import *

import pandas as pd

import numpy as np
from utils import DistanceMetric, Grapher

from .kmeans import KMeans


class BisectingKMeans(object):

    def __init__(self, k, n_iterations=4, dist_metric=DistanceMetric.eucledian, eps=1e-4, do_plot=False):
        """ Initialize Bisecting K-Means Clusterer
        k - number of clusters to fit
        n_iterations - number of iterations of k-means on each cluster division
        dist_metric - the distance metric
        eps - stopping criterion tolerance
        do_plot - voronoi plot after fitting
        """
        self.k = k
        self.n_iterations = n_iterations
        self.dist_metric = dist_metric
        self.eps = eps
        self.do_plot = do_plot
        self.clusters = None
        self.centroids = None
        self.cost = None

    def fit(self, data):
        """ Run the bisecting k-means algorithm
        data - an NxD pandas DataFrame

        returns: a tuple containing
            centroids - a KxD ndarray containing the learned means
            clusters - an N-vector of each point's cluster index
            cost - the total cost of all the points to their assigned cluster
        """

        n, d = data.shape
        clusters = np.zeros(n)
        km = KMeans(2, dist_metric=self.dist_metric, eps=self.eps)
        n_clusters = 1
        while n_clusters < self.k:
            # Find the cluster with most data points and get it's data points
            freq_dist = np.bincount(clusters.astype('int64'))
            divided_cluster_index = np.argmax(freq_dist)
            divided_cluster_data = data[clusters == divided_cluster_index]

            # Run k=2 k-means on the data points n_iterations times and find the best cluster assignment
            least_cost = np.inf
            best_clusters = None
            for _ in range(self.n_iterations):
                km.fit(divided_cluster_data)
                if least_cost > km.cost:
                    least_cost = km.cost
                    best_clusters = km.clusters

            # Put the cluster assignment in the main clusters array
            best_clusters[best_clusters == 1] = n_clusters
            best_clusters[best_clusters == 0] = divided_cluster_index
            np.place(clusters, clusters == divided_cluster_index, best_clusters)
            n_clusters += 1

        # Find the centroids and total cost
        centroids = np.array([np.mean(data[clusters == i], axis=0)
                              for i in range(self.k)])
        cost = np.sum(np.square(data - centroids[clusters.astype('int64')]))

        self.centroids, self.clusters, self.cost = centroids, clusters, cost

        if self.do_plot:
            Grapher().plot_voronoi(data, n_clusters=self.k, clusters=clusters, centroids=centroids,
                                   title="Bisecting K-means clustering with k= " + str(self.k))

        return (centroids, clusters, cost)
