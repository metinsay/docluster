import numpy as np
from collections import Counter
from .preprocessor import Preprocessor


class TfIdf(object):

    def __init__(self, min_df=0.0, max_df=1.0, do_idf=True, preprocessor=Preprocessor()):

        self.min_df = min_df
        self.max_df = max_df
        self.do_idf = do_idf
        self.preprocessor = preprocessor

        self.vocab = None

    def fit(self, documents):
        n_documents = len(documents)
        doc_tfs = []
        df_map = {}

        # Prepare df and tf maps
        for index, document in enumerate(documents):

            tf_map = {}
            doc_tokens = self.preprocessor.fit(document)

            # Each token in the document add the index of document to df, and add 1 to tf
            for token in doc_tokens:
                df_map[token] = set([index]) if token not in df_map else df_map[token] | set([index])
                tf_map[token] = 1 if token not in tf_map else tf_map[token] + 1

            doc_tfs.append(tf_map)

        # Only filter the vocab if necessary
        if self.max_df - self.min_df != 1.0:
            does_token_stay = lambda item: self.min_df <= len(item[1]) / n_documents <= self.max_df
            self.vocab = list(map(lambda item: item[0], filter(does_token_stay, df_map.items())))
        else:
            self.vocab = list(map(lambda item: item[0], df_map.items()))

        # Create vocab_map for easy and fast lookup
        n_vocab = len(self.vocab)
        self.vocab_to_doc = {token: index for index, token in enumerate(self.vocab)}
        tfidf_vector = np.zeros((n_documents, n_vocab))

        # Fill out tfidf_vector
        for doc_id, token_map in enumerate(doc_tfs):
            for token, tf in token_map.items():
                if token in self.vocab_to_doc:
                    token_id = self.vocab_to_doc[token]
                    idf = np.log(n_documents / len(df_map[token])) if self.do_idf else 1
                    tfidf_vector[doc_id, token_id] = tf * idf

        self.tfidf_vector = tfidf_vector
        return tfidf_vector

    def get_values_of_token(self, token):
        token_id = self.vocab_to_doc[token]
        n_documents = self.tfidf_vector.shape[0]
        return np.array([self.tfidf_vector[doc_id, token_id] for doc_id in range(n_documents)])
