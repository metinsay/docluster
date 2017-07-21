import numpy as np
import pandas as pd
import foundation
from enum import Enum
from math import *

class DistanceMetric(enum):
    eucledian = lambda v1, v2 : np.sqrt(np.sum(np.square(v1 - v2), axis=1))
    manhattan = lambda v1, v2 : np.sum(np.absolute(v1 - v2), axis=1)
    angular = lambda v1, v2 : 2 * np.arccos(np.einsum('ij,ij->i', v1, v2) / (np.norm(p1, axis=1) * np.norm(p2, axis=1))) / np.pi

class KMeans(Clusterer):

    def __init__(self, k, dist_func=DistanceMetric.eucledian, eps=1e-4, doGraph=False):
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
        self.clusters = None
        self.centroids = None
        self.cost = None

    def fit(self, data):
        """ Run the k-means algorithm
        data - an NxD pandas DataFrame

        returns: a tuple containing
            centroids - a KxD ndarray containing the learned means
            cluster_assignments - an N-vector of each point's cluster index
            cost - the total cost of all the points to their assigned cluster
        """
        n, d = data.shape
        # randomly choose k points as initial centroids
        centroids = data[random.sample(range(data.shape[0]), self.k)]
        clusters = np.zeros(n)
        prev_cost = -1
        cost = 0
        while abs(cost-prev_cost) > eps:
            prev_cost = cost
            # Assigns every point to the nearest centroid
            for i in range(n):
                tiled_data_point = np.tile(data[i], (self.k,1))
                clusters[i] = np.argmin(self.dist_func(tiled_data_point, centroids))
            cost = 0
            # For every cluster calculates the distance to data points and add to the total cost
            for j in range(self.k):
                data_with_j_assignment = (clusters == j).astype(int)
                centroids[j] = np.dot(data_with_j_assignment, data) / sum(data_with_j_assignment)

                tiled_centroid = np.tile(centroids[j], (len(data),1))
                cost += np.dot(data_with_j_assignment, self.dist_func(data, tiled_centroid))

        self.centroids, self.clusters, self.cost = centroids, clusters, cost

        if self.doGaph:
            self.graph(data, clusters, centroids)

        return (centroids, clusters, cost)

    def graph(self, data, clusters, centroids):
        """ Graph Voronoi Graph
        """
        color_map = {0: "r", 1: "b", 2: "g", 3: "y", 4: "m", 5: "c"}
        colors = [color_map[x % 5] for x in cluster_assign]
        if k > 2:
            voronoi = Voronoi(self.centroids)
            voronoi_plot_2d(voronoi)
        plt.title("K-means clustering with k= " + str(k))
        ax = plt.gcf().gca()
        ax.set_xlim((-15, 5))
        ax.set_ylim((-15, 5))
        plt.scatter(data[:, 0], data[:, 1], color=colors)
        plt.scatter(clusters[:, 0], centroids[:, 1], color="k")
        plt.show()
