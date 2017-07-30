from collections import Counter

import pandas as pd

import numpy as np

from ..constants.file_type import FileType
from ..data_fetcher import FileFetcher
from ..data_saver import FileSaver
from ..visual.grapher import Grapher
from .preprocessor import Preprocessor


class TfIdf(object):

    def __init__(self, n_words=10000, min_df=0.0, max_df=1.0, do_idf=True, preprocessor=Preprocessor(), do_plot=False):
        """ Initialize TfIdf
        n_words - the number of words that is going to be in the vocab, eliminating
                  less frequent words
        min_df - minimum document freqeuncy for a word to be included
        max_df - maximum document frequency for a word to be included
        do_idf - do perform inverse document frequency
        preprocessor - the proproccessor that is going to tokenize the documents
        do_plot - do plot scatter plot after fitting
        """

        self.n_words = n_words
        self.min_df = min_df
        self.max_df = max_df
        self.do_idf = do_idf
        self.preprocessor = preprocessor
        self.do_plot = do_plot

        self.vocab = None
        self.tfidf_vector = None

    def fit(self, documents):
        """ Run Tf-Idf on the documents
        documents - an N list respresenting the documents

        returns:
            tfidf_vector - a NxD ndarray containing the the Tf-Idf vectors with D vocab
        """
        n_documents = len(documents)
        doc_tfs = []
        df_map = {}

        self.document_tokens = []

        # Prepare df and tf maps
        for index, document in enumerate(documents):

            tf_map = {}
            doc_tokens = self.preprocessor.fit(document)
            self.document_tokens.append(doc_tokens)
            # Each token in the document add the index of document to df, and add 1 to tf
            for token in doc_tokens:
                df_map[token] = set(
                    [index]) if token not in df_map else df_map[token] | set([index])
                tf_map[token] = 1 if token not in tf_map else tf_map[token] + 1

            doc_tfs.append(tf_map)

        # Only filter the vocab if necessary
        if self.max_df - self.min_df != 1.0:
            def does_token_stay(item): return self.min_df <= len(
                item[1]) / n_documents <= self.max_df
            self.vocab = list(
                map(lambda item: item[0], filter(does_token_stay, df_map.items())))
        else:
            self.vocab = list(map(lambda item: item[0], df_map.items()))

        # Create vocab_to_doc map for easy and fast lookup
        self.vocab_to_doc = {token: index for index, token in enumerate(self.vocab)}

        n_vocab = len(self.vocab)
        tfidf_vector = np.zeros((n_documents, n_vocab))

        # Fill out tfidf_vector
        for doc_id, token_map in enumerate(doc_tfs):
            for token, tf in token_map.items():
                if token in self.vocab_to_doc:
                    token_id = self.vocab_to_doc[token]
                    idf = np.log(n_documents / len(df_map[token])) if self.do_idf else 1
                    tfidf_vector[doc_id, token_id] = tf * idf

        self.tfidf_vector = tfidf_vector
        self.total_term_freq = np.sum(self.tfidf_vector, axis=0)

        indices = (-self.total_term_freq).argsort()[:self.n_words]
        self.vocab = list(np.take(self.vocab, indices))
        self.tfidf_vector = np.take(self.tfidf_vector, indices)
        if self.do_plot:
            color_assignments = list(
                map(lambda label: 'r' if label == 0 else 'b', documents.index))
            Grapher().plot_scatter(tfidf_vector, color_assignments=color_assignments,
                                   title="Scatter plot of document vectors")

        return tfidf_vector

    def get_values_of_token(self, token, safe=True):
        """ Get the vector of a particular token
        token - a string token
        safe - Check if the token is present in the model

        returns:
            token_vector - a N ndarray containing the  Tf-Idf score
                           of the token with each document
        """
        if not safe or token in self.vocab_to_doc:
            token_id = self.vocab_to_doc[token]
            n_documents = self.tfidf_vector.shape[0]
            return np.array([self.tfidf_vector[doc_id, token_id] for doc_id in range(n_documents)])

    def get_token_vectors(self, do_plot=False):
        """ Get the matrix of token vectors each representing
            the Tf-Idf score of them for each document
        do_plot - do plot scatter plot of token vectors

        returns:
            token_vector - a NxD ndarray containing the token vectors
        """
        n_documents = self.tfidf_vector.shape[0]
        n_vocab = len(self.vocab)
        tokens_vector = np.zeros((n_vocab, n_documents))
        for token in self.vocab:
            token_id = self.vocab_to_doc[token]
            tokens_vector[token_id] = self.get_values_of_token(token, safe=False)

        if do_plot:
            Grapher().plot_scatter(tokens_vector, labels=self.vocab, title="Scatter plot of token vectors")

        return tokens_vector

    def save_model(self, model_name, file_type=FileType.csv, safe=True, directory_path=None):
        """ Save the fitted model
        model_name - the model name / file name
        file_type - the type of file (csv, txt, ...)
        returns:
            token_vector - a NxD ndarray containing the token vectors
        """

        if self.tfidf_vector is None:
            return False
        data = pd.DataFrame(self.tfidf_vector)
        data.columns = self.vocab
        if directory_path:
            file_saver = FileSaver(directory_path=directory_path)
        else:
            file_saver = FileSaver()
        return file_saver.save(data, model_name, file_type=file_type, safe=safe)

    def load_model(self, model_name, file_type=FileType.csv, directory_path=None):
        if directory_path:
            file_fetcher = FileFetcher(directory_path=directory_path)
        else:
            file_fetcher = FileFetcher()
        data = file_fetcher.load(model_name, file_type)
        self.tfidf_vector = data.as_matrix()
        self.vocab = data.columns.tolist()
