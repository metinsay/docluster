import numpy as np

class PCA(DimensionReducer):

    def __init__(self, n_component=2):
        self.n_component = n_component

    def reduce(self, data):
        cov_matrix = np.cov(data)
        eig_values, eig_vectors = cov_matrix.linalg.eig(cov_matrix)
        sorted_eig_indices = eig_values.argsort()[::-1]
        sorted_eig_vectors = eig_vectors[::sorted_eig_indices]
        return np.dot(data, sorted_eig_vectors[:self.n_components])
