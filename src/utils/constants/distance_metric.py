import numpy as np
from enum import Enum
from numpy import *
import numpy as np

class DistanceMetric(Enum):
    eucledian = staticmethod(lambda v1, v2 : sqrt(np.sum(square(array(v1) - array(v2)), axis=1)))
    manhattan = staticmethod(lambda v1, v2 : sum(absolute(v1 - v2), axis=1))
    angular = staticmethod(lambda v1, v2 : 2 * arccos(einsum('ij,ij->i', v1, v2) / (norm(p1, axis=1) * norm(p2, axis=1))) / pi)
