import numpy as np
from collections import Counter

class TfIdf(object):

    def __init__(self, min_df=0.0, max_df=1.0, preprocessor=Preprocessor()):

        self.min_df = min_df
        self.max_df = max_df
        self.preprocessor = preprocessor

    def fit(self, documents):
        n_documents = len(documents)
        labels = []
        doc_maps = {}
        token_to_docs_map = {}
        for index, document, label in enumerate(zip(documents,labels)):
            doc_token_freq_map = {}
            doc_tokens = self.preprocessor.fit(document)

            for token in doc_tokens:

                if token not in token_to_docs_map:
                    token_to_docs_map[token] = set([label])
                elif label not in token_to_docs_map[token]:
                    token_to_docs_map[token].add(label)

                if token not in doc_token_freq_map:
                    doc_token_freq_map[token] = 0
                doc_token_freq_map[token] += 1

            doc_maps[label if label else index] = doc_token_freq_map

        if max_df - min_df != 1.0:
            does_token_stay = lambda token, docs: min_df <= len(docs) / n_documents <= max_df
            vocab = list(map(lambda: token, docs: token, filter(does_token_stay, token_to_docs_map.items())))
        else:
            vocab = list(map(lambda: token, docs: token, token_to_docs_map.items()))
