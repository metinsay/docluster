from enum import Enum

import numpy as np
from numpy import *
from numpy.linalg import norm


class DistanceMetric(Enum):
    eucledian = staticmethod(lambda v1, v2: sqrt(
        np.sum(square(array(v1) - array(v2)), axis=1)))

    manhattan = staticmethod(lambda v1, v2: np.sum(
        absolute(array(v1) - array(v2)), axis=1))

    chebyshev = staticmethod(lambda v1, v2: np.max(
        absolute(array(v1) - array(v2)), axis=1))

    cosine = staticmethod(lambda v1, v2: 1 - (np.sum(array(v1) * array(v2),
                                                     axis=1) / (norm(array(v1), axis=1) * norm(array(v2), axis=1))))

    hamming = staticmethod(lambda v1, v2: np.sum((array(v1) == array(v2)), axis=1))
