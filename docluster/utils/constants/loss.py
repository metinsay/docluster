import numpy as np

class DistanceMetric(Enum):

    def hinge_loss_func(data, weights, offset)
        n_data = len(data)
        labels = data.index
        tiled_weights = np.tile(data, (n_data, 1))
        eval = np.einsum('ij,ij->i', tiled_weights, data) + offset
        mul_label = np.einsum('ij,ij->i', label, eval)
        return np.sum(1 - mul_label.clip(min=0))

    hinge_loss = hinge_loss_func
