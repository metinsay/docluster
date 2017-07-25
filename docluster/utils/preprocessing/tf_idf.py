import numpy as np
from collections import Counter
from .preprocessor import Preprocessor


class TfIdf(object):

    def __init__(self, min_df=0.0, max_df=1.0, preprocessor=Preprocessor()):

        self.min_df = min_df
        self.max_df = max_df
        self.preprocessor = preprocessor

    def fit(self, documents):
        n_documents = len(documents)
        doc_maps = {}
        token_to_docs_map = {}
        for index, document in enumerate(documents):
            doc_token_freq_map = {}
            doc_tokens = self.preprocessor.fit(document)

            for token in doc_tokens:

                if token not in token_to_docs_map:
                    token_to_docs_map[token] = set([index])
                elif index not in token_to_docs_map[token]:
                    token_to_docs_map[token].add(index)

                if token not in doc_token_freq_map:
                    doc_token_freq_map[token] = 0
                doc_token_freq_map[token] += 1

            doc_maps[index if index else index] = doc_token_freq_map

        if self.max_df - self.min_df != 1.0:
            does_token_stay = lambda item: self.min_df <= len(item[1]) / n_documents <= self.max_df
            vocab = list(map(lambda item: item[0], filter(does_token_stay, token_to_docs_map.items())))
        else:
            vocab = list(map(lambda item: item[0], token_to_docs_map.items()))

        n_vocab = len(vocab)
        vocab_map = {token: index for index, token in enumerate(vocab)}
        tfidf_vector = np.zeros((n_documents, n_vocab))
        for doc_id, (document, token_map) in enumerate(doc_maps.items()):
            for token, freq in token_map.items():
                token_id = vocab_map[token]
                tfidf_vector[doc_id, token_id] = freq * np.log(n_documents / len(token_to_docs_map[token]))

        print(tfidf_vector)
