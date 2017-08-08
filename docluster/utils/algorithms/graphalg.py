import random

""" CHECKS """


def check_cycles(graph):
    return False


def check_tree(graph):
    pass


def check_reachability(graph, vertex1, vertex2):
    pass


def check_strong_connectivity(graph):
    pass


""" SEARCH """


def dfs(graph, search):
    pass


def bfs(graph, search):
    pass


""" TRAVERSAL """


def dfs_traverse(graph, source):
    visited = set()

    def dfs_helper(node):
        visited.add(node)
        yield node
        for neighbour in graph.get_neighbours(node):
            if neighbour not in visited:
                yield from dfs_helper(neighbour)

    yield from dfs_helper(source)


def bfs_traverse(graph, source):
    visited = set()
    queue = [source]
    while len(queue) > 0:
        cur_vertex = queue.pop(0)
        visited.add(cur_vertex)
        yield cur_vertex
        for neighbour in graph.get_neighbours(cur_vertex):
            if neighbour not in visited:
                queue.append(neighbour)
                visited.add(neighbour)


""" SORT """


def topological_sort(graph):
    pass


def get_all_topological_sorts(graph):
    pass


""" SHORTEST PATH """


def dijstra(graph, source):
    pass


def bellman_ford(graph, source):
    pass


def floyd_warshall(graph):
    pass


def johnson(graph):
    pass


""" MINIMUM SPANNING TREE """


""" CONNECTIVITY """


def get_disjoint_subgraphs(graph):
    vertices = graph.vertices
    visited = set()
    colorings = {}
    subgraph_index = 0
    while len(vertices) != len(visited):
        unvisited = vertices - visited
        source = random.choice(list(unvisited))
        for vertex in bfs_traverse(graph, source):
            coloring = colorings.get(subgraph_index, set())
            coloring.add(vertex)
            colorings[subgraph_index] = coloring
            visited.add(vertex)

        subgraph_index += 1
    return colorings


""" OTHER """

# http://www.geeksforgeeks.org/find-a-mother-vertex-in-a-graph/


def get_mother_vertex(graph):
    vertices = graph.vertices
    visited = set()
    for source in list(vertices):
        for vertex in dfs_traverse(graph, source):
            print(vertex)
            visited.add(vertex)
            if len(vertices) == len(visited):
                candidate = source
                break
        else:
            continue
        break
    print(visited)
    visited = set()
    for vertex in dfs_traverse(graph, candidate):
        visited.add(vertex)

    return candidate if len(visited) == len(vertices) else None


def d_seperation():
    pass
