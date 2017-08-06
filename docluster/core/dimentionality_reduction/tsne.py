
import numpy as np

from .pca import PCA


class TSNE(object):

    def __init__(self, n_components):
        """
            An implementation of t-Distributed Stochastic Neighbor Embedding.

            Credits:
            --------
            This was adapted from a post/code from Laurens van der Maaten that can be
            forum here:
            https://lvdmaaten.github.io/tsne/

            Paramaters:
            -----------
            n_components : int
                The number of components the data is going to reduced.

            Attributes:
            -----------

        """

        pass

    def fit(self, data):
        """
            Apply t-SNE on the data.

            Paramaters:
            -----------
            data : list(list(float))
                The data that is going to be reduced.

            Return:
            -----------
            reduced_data : list(list(float))
                The data that is reduced to n_components dimentions.
        """

        pass
