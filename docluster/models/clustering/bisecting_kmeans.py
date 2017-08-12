import random
from math import *

import pandas as pd

import numpy as np
from docluster.core import Model
from docluster.utils.constants import DistanceMetric
from docluster.utils.visual import Grapher

from .kmeans import KMeans


class BisectingKMeans(Model):

    def __init__(self, k, n_iterations=4, dist_metric=DistanceMetric.eucledian, eps=1e-4, do_plot=False):
        """
            An implementation of Bisecting K-Means clustering algorithm.

            Paramaters:
            -----------
            k : int
                Number of clusters to fit.
            n_iterations : int
                Number of iterations of regular KMeans with k=2 on each bisection.
            dist_metric : DistanceMetric
                The distance metric which is going to be used to calculates
                distance between vectors.
            eps : float
                Stopping criterion tolerance. This is based on the difference
                between the last epoch and the current epoch.
            do_plot : bool
                If to plot a voronoi diagram of the clusters after fit.

            Attributes:
            -----------
            cost : float
                The sum of distances between centroids and the points belonging
                into the cluster that centroid is defining.
            clusters : list(int)
                Each data points' cluster index starting from 0.
            centroids  : list(list(float))
                The coordinates of each centroid.
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
        """
            Run K-Means on the data.

            Paramaters:
            -----------
            data : list(list(float))
                The data that is going to be clustered.

            Return:
            -------
            clusters : list(int)
                Each data points' cluster index starting from 0.
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

        return clusters
