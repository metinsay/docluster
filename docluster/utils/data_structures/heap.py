
class Heap(object):

    def __init__(self):
        self.root = None
        self.data = [None]

    def insert(self, key, value=None):

        self.data += [(key, value)]
        index = len(self.data) - 1
        return self._trickle_up(index)

    def delete(self, index):
        if index > len(self.data) - 1 or len(data) < 2:
            raise ValueError('Out of of bounds.')

        self.data[index] = self.data[-1]
        del self.data[-1]

    def get_max(self):
        return self.data[1] if len(self.data) > 1 else None

    def pop_max(self):
        # TODO: Implement Heap Pop Max
        pass

    def _get_parent_index(self, index):
        return index // 2

    def _get_parent(self, index):
        return self.data[self._get_parent_index(index)]

    def _trickle_down(self, index):
        node = self.data[index]

    def _trickle_up(self, index):
        node = self.data[index]
        parent_index = self._get_parent_index(index)
        parent = self._get_parent(index)
        if parent and parent[0] < node[0]:
            self.data[index], self.data[parent_index] = parent, node
            self._trickle_up(parent_index)

        return self.data

    def push(self):
        pass

    def pop(self):
        pass
