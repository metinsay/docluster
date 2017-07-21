import numpy as np

class DistanceMetric(enum):
    eucledian = lambda v1, v2 : np.sqrt(np.sum(np.square(v1 - v2), axis=1))
    manhattan = lambda v1, v2 : np.sum(np.absolute(v1 - v2), axis=1)
    angular = lambda v1, v2 : 2 * np.arccos(np.einsum('ij,ij->i', v1, v2) / (np.norm(p1, axis=1) * np.norm(p2, axis=1))) / np.pi
