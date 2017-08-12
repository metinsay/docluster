class Model(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.func = lambda: None

    def __init__(self, func, model_name):
        self.model_name = model_name
        self.func = func

    def fit(self, data):
        return self.func()

    def __str__(self):
        return self.model_name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return False  # fixme

    def __hash__(self):
        return 3  # fixme
