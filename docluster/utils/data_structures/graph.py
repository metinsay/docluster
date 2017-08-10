class Edge(object):

    def __init__(self, start_node, end_node, weight=None):
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
        self.neighbours = {}
        self.edges = set()
        self.vertices = set()

    def link(self, vertex1, vertex2, weight=None, override=False, directed=True):

        self.vertices.add(vertex1)
        self.vertices.add(vertex2)
        new_edges = set([Edge(vertex1, vertex2, weight)])
        vertex1_set = self.neighbours.get(vertex1, set())
        vertex1_set.add(vertex2)
        self.neighbours[vertex1] = vertex1_set
        if not directed:
            new_edges.add(Edge(vertex2, vertex1, weight))
            vertex2_set = self.neighbours.get(vertex2, set())
            vertex2_set.add(vertex1)
            self.neighbours[vertex2] = vertex2_set
        if override:
            # TODO: Unlink the vertices
            pass

        self.edges |= new_edges

    def unlink(self, vertex1, vertex2, remove_aliens=False, safe=True):
        pass

    def subgraph_with_vertices(self, vertices):
        g = Graph()
        for vertex in list(vertices):
            if vertex in self.neighbours:
                for neighbour in list(self.neighbours[vertex]):
                    g.link(vertex, neighbour)
        return g

    def get_neighbours(self, vertex, safe=True):
        if not safe or vertex in self.neighbours:
            return list(self.neighbours[vertex])
        return []

    def __len__(self):
        return len(self.vertices)

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
