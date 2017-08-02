import collections
import math
import multiprocessing
import os
import random
import threading

import pandas as pd

import deepcopy
import numpy as np
import tensorflow as tf
from scipy.special import expit
from utils import DistanceMetric, FileFetcher, FileSaver, FileType

from ..document_embedding import TfIdf
from ..preprocessing import Preprocessor, TokenFilter
from .word_embeddings import WordEmbeddings


class Word2Vec(object):

    def __init__(self, preprocessor=None, n_skips=16, n_negative_samples=100, n_words=10000, embedding_size=100, batch_size=32, window_size=10, learning_rate=0.025, n_epochs=1, n_workers=4, do_plot=False):
        """
            A Skip-Gram model Word2Vec with multi-thread training capability.

            Paramaters:
            -----------
            preprocessor : Preprocessor
                The preprocessor that will tokenize the documents.
                The default one also filters punctuation, tokens with numeric
                characters and one letter words. Furthermore, no stemming or
                lemmatization is applied. All these can be adjusted
                by passing a custom preprocessor.
            n_skip : int
                The number of skips.
            n_negative_samples : int
                The number of negative samples that are going to collected for each
                batch.
            n_words : int
                The number of words that the vocabulary will have. The filtering is
                based on the word frequency. Therefore, less frequent words will not
                be included in the vocabulary.
            embedding_size : int
                The size of the embedding vectors. Usually the more makes the embeddings
                more accurate, but this is not always the case. Increasing the size
                dramatically affects trainning time.
            batch_size : int
                The batch size.
            window_size : int
                The window size where the words to the left and to the right of the words
                will give context to the word.
            learning_rate : int
                The initial learning rate of the gradient decent.
            n_epochs : int
                The number of epoches the model is going to be trained. Increasing the number
                dramatically affects trainning time.
            n_workers : int
                The number of workers that is going to train the model concurrently.
                It is not recommended to use more than the number of core.
            do_plot : bool

            Attributes:
            -----------
            embeddings :
                The embedding vectors that represents each word
        """
        if preprocessor is None:
            additional_filters = [lambda token: len(token) == 1]
            token_filter = TokenFilter(filter_stop_words=False,
                                       additional_filters=additional_filters)
            preprocessor = Preprocessor(do_stem=False, do_lemmatize=False,
                                        parse_html=False, token_filter=token_filter, lower=False)

        self.preprocessor = preprocessor
        self.n_skips = n_skips
        self.n_negative_samples = n_negative_samples
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_words = n_words
        self.n_workers = n_workers

        self._total_loss = 0
        self._dist_metric = DistanceMetric.cosine
        self.embeddings = WordEmbeddings(size=embedding_size, n_words=n_words)
        self.locks = np.ones(n_words)

        self.syn1 = np.zeros((n_words, embedding_size))
        self.syn1neg = np.zeros((n_words, embedding_size))

    def fit(self, documents):
        """
            Train the Word2Vec model with the documents.

            Paramaters:
            -----------
            documents : list(str)
                the documents that the Word2Vec model is going to learn the embeddings from.
        """
        n_words_trained = 0
        tokens, self.vocab, data, self._frequencies, self.diction, self.reverse_diction = self._build_dataset(
            documents)
        n_tokens = len(tokens)
        n_vocab = len(self.vocab)
        words_per_epoch = n_vocab / self.n_epochs
        self._cum_dist = self._build_cum_dist()

    def _build_dataset(self, documents):
        """Preprocesses the documents and creates the dataset for fitting."""

        # Get the term frequencies without idf
        tfidf = TfIdf(do_idf=False, preprocessor=self.preprocessor, n_words=self.n_words)
        tfidf.fit(documents)

        # Flatten the document tokens to create one long list
        tokens = list(np.hstack(np.array(tfidf.document_tokens)))

        # Create the vocab list with 'UNK' for vocab that couldn't make the vocab list
        vocab = tfidf.vocab
        vocab_set = set(vocab)

        diction = {token: index for index, token in enumerate(vocab)}
        reverse_diction = dict(zip(diction.values(), diction.keys()))

        # Turn the long token list into a index references to the diction
        data = list(map(lambda token: diction[token]
                        if token in vocab_set else 0, tokens))

        # Get the frequencies of tokens and add the frequency of 'UNK' at the beginning
        # frequencies = np.insert(tfidf.total_term_freq, 0, data.count(0))[:self.n_words]
        frequencies = tfidf.total_term_freq[:self.n_words]

        return tokens, vocab, data, frequencies, diction, reverse_diction

    def _build_cum_dist(self, distortion=0.75, domain=2**31 - 1):

        freq_total = np.sum(self._frequencies ** distortion)
        cum_dist = np.cumsum(self._frequencies) * domain / freq_total
        return cum_dist

    def _train(self, data, optimizer, loss):
        """Train the model."""

        start_index = 0
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:

            self._sess = sess
            self._sess.run(init_op)
            for epoch in range(self.n_epochs):
                self._train_one_epoch(data, optimizer, loss)

                print("Epoch:", (epoch + 1))

            self.embeddings = self._embeddings.eval()

        print("\nTraining complete!")

    def _train_one_example(self, example, label, alpha):

        predict_word = model.wv.vocab[word]  # target word (NN output)

        # input word (NN input/projection layer)
        example_index = self._diction[example]
        embedding = self.embeddings.vectors[example_index]
        lock = self.locks[example_index]

        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        # 2d matrix, codelen x layer1_size
        l2a = deepcopy(self.syn1[predict_word.point])
        prod_term = np.dot(embedding, l2a.T)
        fa = expit(prod_term)  # propagate hidden -> output
        # vector of error gradients multiplied by the learning rate
        ga = (1 - predict_word.code - fa) * alpha
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output

        sgn = (-1.0)**predict_word.code  # `ch` function, 0 -> 1, 1 -> -1
        lprob = -log(expit(-sgn * prod_term))
        self._total_loss += sum(lprob)

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            # vector of error gradients multiplied by the learning rate
            gb = (model.neg_labels - fb) * alpha
            if learn_hidden:
                model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output

            # loss component corresponding to negative sampling
            if compute_loss:
                # for the sampled words
                self._total_loss -= sum(log(expit(-1 * prod_term[1:])))
                # for the output word
                self._total_loss -= log(expit(prod_term[0]))

        if learn_vectors:
            # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
            embedding += neu1e * lock_factor

    def _train_one_epoch(self, data, optimizer, loss):
        """Train one epoch with workers."""
        # Each worker generates a batch and trains it until posion pill

        def worker_duty():
            """The duty of a single worker."""

            while True:
                batch = queue.get()
                if batch is None:
                    break
                examples, labels, alphas = batch
                for example, label, alpha in batch:
                    self._train_one_example(example, label, alpha)

        def generate_batch():
            """Create a batch for a training step in Word2Vec."""

            # Initialize variables
            example = np.zeros(self.batch_size)
            labels = np.zeros((self.batch_size, 1))
            alphas = np.zeros(self.batch_size)
            n_items = 0
            index = 0

            while index < len(data):
                reduced_window = random.randint(0, self.window_size)
                if data[index] not is None:

                    left = max(0, index - self.window_size + reduced_window)
                    right = min((index + self.window_size + 1 -
                                 reduced_window), len(data) - 1)
                    for pos2 in range(left, right, 1):

                        if n_items == self.batch_size:
                            queue.put((example, labels, index))
                            example = np.zeros(self.batch_size)
                            labels = np.zeros((self.batch_size, 1))
                            n_items = 0

                        if pos2 != index and data[pos2] not is None:
                            example[n_items] = data[pos2]
                            labels[n_items] = data[index]
                            alpha = self.learning_rate - \
                                (self.learning_rate - 0.001) * (index / self.n_words)
                            alphas[n_items] = max(0.001, alpha)
                            n_items += 1
                index += 1

            # Poison pills
            for _ in range(n_workers):
                queue.put(None)

        # Create a threadsafe queue to store the batch indexes
        queue = multiprocessing.Queue(maxsize=2 * self.n_workers)

        # Create and run the threads
        workers = [threading.Thread(target=generate_batch)]
        workers.extend([threading.Thread(target=worker_duty) _ in range(self.n_workers - 1)])

        for worker in workers:
            worker.start()

        for thread in workers:
            thread.join()

    def most_similar_words(self, word, n_words=5, include_similarity=False):
        """
            Get the most similar words to a word.

            Paramaters:
            -----------
            word : list(str)
                The word that is the point of intrest.
            n_words : int
                The number of words that is going to be returned.
            include_similarity : bool
                If to include the similarity score as part of a tuple next to the words.

            Return:
            -------
            similar_words : list(str) or list(tuple(str, float))
                The words that are most similar to the word according to the trained
                embeddings.
        """

        if word in self.vocab:
            token_id = self.diction[word]
            tiled_embedding = np.tile(self.embeddings[token_id], (self.n_words, 1))
            embedding_similarities = self._dist_metric(tiled_embedding, self.embeddings)
            most_similar_token_ids = (-embedding_similarities).argsort()

            return list(map(lambda token_id: self.reverse_diction[token_id], most_similar_token_ids))
        else:
            print('not in vocab')

    def save_model(self, model_name, file_type=FileType.csv, safe=True, directory_path=None):
        """
            Save the fitted model.

            Paramaters:
            -----------
            model_name : str
                The model name (also the file name) of the model is going to be saved under.
            file_type : FileType
                The file type that the model is going to be saved as.

            Return:
            -------
            saved : bool
                If the model is saved successfully or not.
        """

        if self.embeddings is None:
            return False

        data = pd.DataFrame(self.embeddings.T)
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

        self.n_words += 1
        data = file_fetcher.load(model_name, file_type)

        self.embeddings = data.as_matrix().T
        self.vocab = data.columns.tolist()
        self.diction = {token: index for index, token in enumerate(self.vocab)}
        self.reverse_diction = dict(zip(self.diction.values(), self.diction.keys()))
