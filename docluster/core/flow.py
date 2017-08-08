import pandas as pd

from utils import Graph, Timer, graphalg, seqeuencealg


class Flow(object):

    def __init__(self, do_thread_braches=True, do_analytics=True, show_progress=True):
        self.graph = Graph()
        self.do_analytics = do_analytics
        self.show_progress = show_progress

    def fit(self, data, m):
        if graphalg.check_cycles(self.graph):
            raise ValueError("The flow shouldn't have loops")

        prevData = data

        models_in_execution_order = []
        times = []
        print(self.graph)
        for model in graphalg.bfs_traverse(self.graph, m):
            print(model)
            if self.do_analytics:
                with Timer() as t:
                    prevData = model.fit(prevData)

                models_in_execution_order.append(model)
                times.append(t.interval)
            else:
                prevData = model.fit(prevData)

        if self.do_analytics:
            analytics = pd.DataFrame({'Time (s)': times})
            analytics.index = models_in_execution_order
            print(analytics)
        return prevData

    def link(self, start_model, end_model):
        self.graph.link(start_model, end_model, override=True, directed=True)

    def chain(self, *models):
        for start_model, end_model in seqeuencealg.n_gram(models, 2):
            self.graph.link(start_model, end_model, override=True, directed=True)