import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


class Grapher(object):

    def __init__(self, colors=[], ):
        """ Initialize Grapher
        colors - an N-vector with colors
        """
        self.colors = colors

    def plot_voronoi(self, data, clusters, centroids, title, dim_reducer=PCA()):
        """ Graph Voronoi Graph
        data - an NxD pandas DataFrame
        clusters - an N-vector with each point's cluster index
        centroids - a KxD ndarray containing the learned means
        """
        reduced_data = dim_reducer.reduce(data, n_component=2)

        colors_assigns= [self.colors[x % len(self.colors)] for x in clusters]
        voronoi_plot_2d(Voronoi(self.centroids))
        plt.title(title)
        ax = plt.gcf().gca()
        ax.set_xlim((-15, 5))
        ax.set_ylim((-15, 5))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color=colors_assigns)
        plt.scatter(clusters[:, 0], centroids[:, 1], color="k")
        plt.show()
