class Edge(object):

    def __init__(self, start_node, end_node, weight=0):
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

    def __eq__(self, other):
        return type(other) == type(self) and other.start_node == self.start_node and other.end_node == self.end_node and other.weight == self.weight

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '(' + str(self.start_node) + ', ' + str(self.end_node) + ', ' + str(self.weight) + ')'

    def __hash__(self):
        return hash(self.start_node) + hash(self.end_node) + hash(self.weight)


class Graph(object):

    def __init__(self):
        self.edges = set()
        self.vertices = set()

    def link(self, vertex1, vertex2, weight=0, override=False, directed=False):

        self.vertices.add(vertex1)
        self.vertices.add(vertex2)
        new_edges = set([Edge(vertex1, vertex2, weight)])
        if not directed:
            new_edges.add(Edge(vertex2, vertex1, weight))

        if override:
            # TODO: Unlink the vertices
            pass

        self.edges |= new_edges

    def unlink(self, vertex1, vertex2, remove_aliens=False, safe=True):
        pass

    def get_neighbours(self, vertex):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '{'
        for i, edge in enumerate(self.edges):
            s += str(edge)

            if i != len(self.edges) - 1:
                s += ', '
        s += '}'
        return s
