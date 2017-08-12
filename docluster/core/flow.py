import pandas as pd

import numpy as np


class Flow(object):

    def __init__(self, do_thread_braches=True, do_analytics=True, show_progress=True):
        self.graph = Graph()
        self.do_analytics = do_analytics
        self.show_progress = show_progress

    def fit(self, data, m):
        if graphalg.check_cycles(self.graph):
            raise ValueError("The flow shouldn't have loops")

        # branches = graphalg.get_disjoint_subgraphs(self.graph)
        # print(branches, 'a')
        # for branch in branches:
        #     print(graphalg.get_mother_vertex(branch), 'b')
        # return
        prevData = data

        models = []
        times = []
        shapes = []

        for model in graphalg.bfs_traverse(self.graph, m):
            if self.do_analytics:
                with Timer() as t:
                    prevData = model.fit(prevData)

                models.append(model)
                times.append(t.interval)
                shapes.append(np.array(prevData).shape)
            else:
                prevData = model.fit(prevData)

        if self.do_analytics:
            analytics = pd.DataFrame({'Time (s)': times, 'Return shape': shapes})
            analytics.index = models
            print(analytics)

        return prevData

    def link(self, start_model, end_model):
        self.graph.link(start_model, end_model, override=True, directed=True)

    def chain(self, *models):
        if len(models) < 2:
            raise ValueError("Chaining needs at least two models.")

        for start_model, end_model in seqeuencealg.n_gram(models, 2):
            self.graph.link(start_model, end_model, override=True, directed=True)
