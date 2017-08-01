import collections
import math
import multiprocessing
import os
import random
import threading

import pandas as pd

import numpy as np
import tensorflow as tf

from ..constants.distance_metric import DistanceMetric
from ..constants.file_type import FileType
from ..data_fetcher import FileFetcher
from ..data_saver import FileSaver
from .preprocessor import Preprocessor
from .tf_idf import TfIdf
from .token_filter import TokenFilter


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

        self._dist_metric = DistanceMetric.cosine

    def fit(self, documents):
        """
            Train the Word2Vec model with the documents.

            Paramaters:
            -----------
            documents : list(str)
                the documents that the Word2Vec model is going to learn the embeddings from.
        """
        n_words_trained = 0
        tokens, self.vocab, data, frequencies, self.diction, self.reverse_diction = self._build_dataset(
            documents)
        n_tokens = len(tokens)
        n_vocab = len(self.vocab)
        words_per_epoch = n_vocab / self.n_epochs
        batch_logit, negative_samples_logit = self._build_graph(data, frequencies)
        loss = self._build_loss_metric(batch_logit, negative_samples_logit)
        optimizer = self._build_optimizer(loss, words_per_epoch, n_words_trained)
        self._train(data, optimizer, loss)

    def _build_dataset(self, documents):
        """Preprocesses the documents and creates the dataset for fitting."""

        # Get the term frequencies without idf
        tfidf = TfIdf(do_idf=False, preprocessor=self.preprocessor, n_words=self.n_words)
        tfidf.fit(documents)

        # Flatten the document tokens to make them on long list
        tokens = list(np.hstack(np.array(tfidf.document_tokens)))

        # Create the vocab list with 'UNK' for  vocab that couldn't make the vocab list
        vocab = ['UNK'] + tfidf.vocab
        vocab_set = set(vocab)
        self.n_words += 1
        diction = {token: index for index, token in enumerate(vocab)}
        reverse_diction = dict(zip(diction.values(), diction.keys()))

        # Turn the long token list into a index references to the diction
        data = list(map(lambda token: diction[token]
                        if token in vocab_set else 0, tokens))

        # Get the frequencies of tokens and add the frequency of 'UNK' at the beginning
        frequencies = np.insert(tfidf.total_term_freq, 0, data.count(0))[:self.n_words]

        return tokens, vocab, data, frequencies, diction, reverse_diction

    def _build_graph(self, data, frequencies):
        """Build the graph that is going to be trained."""

        self._batch_input = tf.placeholder(tf.int64, shape=[self.batch_size])
        self._labels_input = tf.placeholder(tf.int64, shape=[self.batch_size, 1])

        width = 0.5 / self.embedding_size

        embeddings_dim = [self.n_words, self.embedding_size]
        self._embeddings = tf.Variable(tf.random_uniform(
            embeddings_dim, minval=-width, maxval=width), name="embeddings")

        batch_embeddings = tf.nn.embedding_lookup(self._embeddings, self._batch_input)

        softmax_weights = tf.Variable(tf.zeros(embeddings_dim), name="softmax_weights")
        softmax_biases = tf.Variable(tf.zeros([self.n_words]), name="softmax_biases")

        label_weights = tf.nn.embedding_lookup(softmax_weights, self._labels_input)
        label_biases = tf.nn.embedding_lookup(softmax_biases, self._labels_input)

        negative_sample_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(true_classes=self._labels_input, num_true=1,
                                                                          num_sampled=self.n_negative_samples, unique=True,
                                                                          range_max=self.n_words, distortion=0.75, unigrams=list(frequencies))

        negative_sample_weights = tf.nn.embedding_lookup(
            softmax_weights, negative_sample_ids)
        negative_sample_biases = tf.nn.embedding_lookup(
            softmax_biases, negative_sample_ids)

        batch_logit = tf.reduce_sum(tf.multiply(
            batch_embeddings, label_weights), 1) + label_biases

        reduced_negative_samples = tf.reshape(
            negative_sample_biases, [self.n_negative_samples])
        negative_samples_logit = tf.matmul(
            batch_embeddings, negative_sample_weights, transpose_b=True) + reduced_negative_samples

        return batch_logit, negative_samples_logit

    def _build_loss_metric(self, batch_logit, negative_samples_logit):
        """Build loss metric that will be optimized."""

        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(batch_logit), logits=batch_logit)
        negative_samples_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(negative_samples_logit), logits=negative_samples_logit)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        loss = (tf.reduce_sum(batch_loss) +
                tf.reduce_sum(negative_samples_loss)) / self.batch_size
        return loss

    def _build_optimizer(self, loss, words_per_epoch, n_words_trained):
        """Build the optimizer of the loss."""

        n_words_to_train = float(words_per_epoch * self.n_epochs)
        learning_rate = self.learning_rate * \
            tf.maximum(0.0001, 1.0 - tf.cast(n_words_trained,
                                             tf.float32) / n_words_to_train)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = optimizer.minimize(loss, global_step=tf.Variable(0, name="global_step"),
                                       gate_gradients=optimizer.GATE_NONE)
        return optimizer

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

    def _train_one_epoch(self, data, optimizer, loss):
        """Train one epoch with workers."""
        # Each worker generates a batch and trains it until posion pill

        def worker_duty(q):
            while True:
                batch = q.pop(0)
                if batch is None:
                    break
                example, labels, index = batch
                _, error = self._sess.run([optimizer, loss], feed_dict={
                    self._batch_input: example, self._labels_input: labels})

        # Create a threadsafe queue to store the batch indexes
        queue = []
        for batch in self._generate_batch(data):
            queue.append(batch)

        # Poison pills
        for _ in range(self.n_workers):
            queue.append(None)

        # Create and run the threads
        workers = []
        for _ in range(self.n_workers):
            thread = threading.Thread(target=worker_duty, kwargs={'q': queue})
            thread.start()
            workers.append(thread)

        for thread in workers:
            thread.join()

    def _generate_batch(self, data):
        """Create a batch for a training step in Word2Vec."""

        # Initialize variables
        example = np.zeros(self.batch_size)
        labels = np.zeros((self.batch_size, 1))
        n_items = 0
        index = 0
        while index < len(data):
            # `b` in the original word2vec code
            reduced_window = random.randint(0, self.window_size)

            # now go over all words from the (reduced) window, predicting each one in turn
            left = max(0, index - self.window_size + reduced_window)
            for pos2 in range(left, min((index + self.window_size + 1 - reduced_window), len(data) - 1), 1):
                    # don't train on the `word` itself
                if n_items == self.batch_size:
                    yield example, labels, index
                    example = np.zeros(self.batch_size)
                    labels = np.zeros((self.batch_size, 1))
                    n_items = 0

                if pos2 != index:
                    example[n_items] = data[index]
                    labels[n_items] = data[pos2]
                    n_items += 1
            index += 1

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
