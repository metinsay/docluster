
import numpy as np

from .pca import PCA


class TSNE(object):

    def __init__(self, n_components=2, n_initial_reduction=50, perplexity=30.0, n_epochs=1000, momentum_range=(0.5, 0.8)):
        """
            An implementation of t-Distributed Stochastic Neighbor Embedding.

            Credits:
            --------
            This was adapted from a post/code from Laurens van der Maaten that can be
            found here:
            https://lvdmaaten.github.io/tsne/

            Paramaters:
            -----------
            n_components : int
                The number of components the data is going to reduced.

            Attributes:
            -----------

        """

        self.n_components = n_components
        self.perplexity = perplexity
        self.n_initial_reduction = n_initial_reduction
        self.n_epochs = n_epochs
        self.inititial_momentum, self.final_momentum = momentum_range

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
        reduced_data = PCA(self.n_initial_reduction).fit(data)
        p_values = self._calculate_p_values(data)

    def _calculate_p_values(self, data):
        """Calculates the P-values."""

        pass
