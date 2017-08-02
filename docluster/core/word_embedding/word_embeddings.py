import numpy as np


class WordEmbeddings(object):

    def __init__(self, size=300, n_words=10000):
        self.size = size
        self.n_words = n_words
        width = 0.5 / self.size
        self.embeddings = np.random.uniform(min=-width, max=width, size=(n_words, size))
