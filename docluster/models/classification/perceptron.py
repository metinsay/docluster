import numpy as np


class Perceptron(object):

    def __init__(self, n_iterations=10, kind='standard'):
        self.n_iterations = n_iterations
        self.kind = kind

    def train(self, data, labels):
        n_data, n_features = data.shape
        weights = np.zeros(n_features)
        offset = 0

        if self.kind == 'standard':

            for _ in range(self.n_iterations):
                for feature, label in zip(data, labels):
                    (weights, offset) = self._update_weights(
                        weights, offset, feature, label)

            self.weights, self.offset = weights, offset

        elif self.kind == 'average':

            sum_weights = np.zeros(n_features)
            sum_offset = 0.

            for _ in range(self.n_iterations):
                for feature_vector, label in zip(data, labels):
                    (weights, offset) = self._update_weights(
                        feature_vector, label, weights, offset)
                    sum_theta = np.add(sum_theta, weights)
                    sum_offset += offset

            self.weights, self.offset = sum_theta / \
                (n_data * self.n_iterations), sum_offset / (n_data * self.n_iterations)

        else:
            None  # TODO: Give error

        return self.weights, self.offset

    def _update_weights(self, weights, offset, feature, label):
        if label * (np.dot(feature, weights) + offset) <= 0:
            weights = np.add(weights, label * feature)
            offset = offset + label
        return (weights, offset)

    def fit(self, data):
        n_data = data.shape[0]

        tiled_weights = np.tile(self.weights, (n_data, 1))
        print(tiled_weights, data)
        evaled_data = np.einsum('ij,ij->i', tiled_weights, data) + self.offset
        return (evaled_data <= 0).astype('int64')
