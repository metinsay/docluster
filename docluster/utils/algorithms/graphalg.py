""" CHECKS """


def check_cycles(graph):
    return False


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


""" SHORTEST PATH """


def dijstra(graph, source):
    pass


def bellman_ford(graph, source):
    pass


def floyd_warshall(graph):
    pass


def johnson(graph):
    pass


""" CONNECTIVITY """
