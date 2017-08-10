import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


class Grapher(object):

    def __init__(self, colors=["r", "b", "g", "y", "m", "c"]):
        """ Initialize Grapher
        colors - an N-vector with colors
        """
        self.colors = colors

    def plot_scatter(self, data, labels=[], color_assignments=[], title=''):

        data = self.reduce_data(data)
        fig, ax = plt.subplots()
        ax.scatter(data[:, 0], data[:, 1], color=color_assignments)

        texts = []
        for i, txt in enumerate(labels):
            ax.text(data[:, 0][i], data[:, 1][i], txt)

        plt.title(title)
        plt.show()

    def plot_voronoi(self, data, n_clusters, clusters, centroids, title):
        """ Graph Voronoi Graph
        data - an NxD pandas DataFrame
        clusters - an N-vector with each point's cluster index
        centroids - a KxD ndarray containing the learned means
        """
        data = self.reduce_data(data)
        centroids = self.reduce_data(centroids)

        colors_assigns = [self.colors[int(x) % len(self.colors)] for x in clusters]
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

    def reduce_data(self, data):
        # Reduce the dimension to 2D with PCA

        return data
