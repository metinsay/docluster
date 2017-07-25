import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from ..dimension_reduction.pca import PCA

class Grapher(object):

    def __init__(self, colors=["r", "b", "g", "y", "m", "c"]):
        """ Initialize Grapher
        colors - an N-vector with colors
        """
        self.colors = colors

    def plot_voronoi(self, data, n_clusters, clusters, centroids, title):
        """ Graph Voronoi Graph
        data - an NxD pandas DataFrame
        clusters - an N-vector with each point's cluster index
        centroids - a KxD ndarray containing the learned means
        """

        # Reduce the dimension to 2D with PCA
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit(data)
            centroids = pca.reduce(centroids)

        colors_assigns= [self.colors[int(x) % len(self.colors)] for x in clusters]
        if n_clusters > 2:
            voronoi_plot_2d(Voronoi(centroids))
        plt.title(title)

        plt.scatter(data[:, 0], data[:, 1], color=colors_assigns)
        plt.scatter(centroids[:, 0], centroids[:, 1], color="k")
        plt.show()

    def plot_heat_map(self, data, title):
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(title)
        plt.show()
