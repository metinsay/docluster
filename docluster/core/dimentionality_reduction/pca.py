import numpy as np
from docluster.core import Model


class PCA(Model):

    def __init__(self, n_components=2, model_name='PCA'):
        """
            An implementation of Principal Components Analysis.

            Paramaters:
            -----------
            n_components : int
                The number of components the data is going to reduced.

            Attributes:
            -----------
            eig_vectors : list(list(float))
                The eigenvectors of the data.
        """

        self.n_components = n_components
        self.eig_vectors = None
        self.model_name = model_name

    def fit(self, data):
        """
            Apply PCA on the data.

            Paramaters:
            -----------
            data : list(list(float))
                The data that is going to be reduced.

            Return:
            -----------
            reduced_data : list(list(float))
                The pricipal components of data reduced to the n_components.
        """
        # Estimate the covariance matrix
        n_points = len(data)
        mean = np.tile(np.mean(data, axis=0), (n_points, 1))
        centered_data = data - mean
        scatter_matrix = np.dot(centered_data.T, centered_data)

        # Find the eigenvectors with maximum eigenvalue
        eig_values, eig_vectors = np.linalg.eig(scatter_matrix)
        sorted_eig_indices = eig_values.argsort()[::-1]
        sorted_eig_vectors = eig_vectors[:, sorted_eig_indices]
        self.eig_vectors = sorted_eig_vectors[:, range(self.n_components)]
        # Matrix multiple the eigenvector with the data
        return np.dot(centered_data, self.eig_vectors)
