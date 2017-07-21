import numpy as np

class PCA(DimensionReducer):

    def __init__(self, n_component=2):
        self.n_component = n_component

    def reduce(self, data):
        pass
