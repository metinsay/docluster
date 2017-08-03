
class BSTNode(object):

    def __init__(self, key, value, left_child=None, right_child=None, parent=None):

        self.key = key
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

    @property
    def is_root(self):
        return not self.parent

    @property
    def is_leaf(self):
        return not self.left_child and not self.right_child

    @property
    def has_right_child(self):
        return self.right_child

    @property
    def has_left_child(self):
        return self.left_child

    def __str__(self):
        return str((self.key, self.value))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        if self:
            if self.has_left_child:
                for left_child in self.left_child:
                    yield left_child
            yield self
            if self.has_right_child:
                for right_child in self.right_child:
                    yield right_child


class BST(object):

    def __init__(self):
        self.size = 0
        self.root = None

    def __len__(self):
        return self.size

    def __contains__(self, key):
        try:
            self.get(key)
        except Exception as e:
            return False
        else:
            return True

    def __str__(self):
        return str(self.in_order_traverse())

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        if self.root:
            for key in self.root:
                yield key
        else:
            raise ValueError("No values inside BST.")

    def insert(self, key, value):

        def put(node):
            if key < node.key:
                if node.has_left_child:
                    put(node.left_child)
                else:
                    node.left_child = BSTNode(key, value, parent=node)
            else:
                if node.has_right_child:
                    put(node.right_child)
                else:
                    node.right_child = BSTNode(key, value, parent=node)

        if self.root:
            put(self.root)
        else:
            self.root = BSTNode(key, value)
        self.size += 1

    def get(self, key):
        if not self.root:
            raise ValueError('No values inside BST.')
        cur_node = self.root
        while cur_node.key != key:
            if cur_node.is_leaf:
                raise ValueError('Value not in BST.')
            if key < cur_node.key:
                cur_node = cur_node.left_child
            else:
                cur_node = cur_node.right_child
        return cur_node

    def delete(self):
        pass

    def get_min(self):
        cur_node = self
        while cur_node.left_child:
            cur_node = cur_node.left_child
        return cur_node

    def get_max(self):
        cur_node = self
        while cur_node.right_child:
            cur_node = cur_node.right_child
        return cur_node

    def in_order_traverse(self):
        return [key for key in self]
