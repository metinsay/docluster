import numpy as np


class PCA(DimensionReducer):

    def __init__(self, n_components=2):
        """ Initialize PCA Dimension Reducer
        n_components - number of components the data is going to reduced
        """
        self.n_components = n_components
        self.eig_vectors = None

    def fit(self, data):
        """ Apply PCA on the data
        data - an NxD pandas DataFrame

        returns:
            reduced_data - a Nxn_components ndarray that represents
                           the pricipal components of data

        """
        # Estimate the covariance matrix
        mean = np.mean(data, axis=0)
        centered_data = data - mean
        scatter_matrix = np.dot(centered_data.T, centered_data)

        # Find the eigenvectors with maximum eigenvalue
        eig_values, eig_vectors = np.linalg.eig(scatter_matrix)
        sorted_eig_indices = eig_values.argsort()[::-1]
        sorted_eig_vectors = eig_vectors[:, sorted_eig_indices]
        self.eig_vectors = sorted_eig_vectors[:, range(self.n_components)]

        # Matrix multiple the eigenvector with the data
        return np.dot(centered_data, self.eig_vectors)
