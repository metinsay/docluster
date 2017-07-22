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
        pca = PCA(n_components=2)
        reduced_data = pca.reduce(data)
        reduced_centroids = pca.reduce(centroids)

        colors_assigns= [self.colors[int(x) % len(self.colors)] for x in clusters]
        if n_clusters > 2:
            voronoi_plot_2d(Voronoi(reduced_centroids))
        plt.title(title)
        ax = plt.gcf().gca()

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color=colors_assigns)
        plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], color="k")
        plt.show()
