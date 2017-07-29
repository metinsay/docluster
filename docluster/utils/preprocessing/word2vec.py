import tensorflow as tf
import os
import collections
from .preprocessor import Preprocessor
from .token_filter import TokenFilter
from .tf_idf import TfIdf
import numpy as np
import random
import math
class Word2Vec(object):

    def __init__(self, n_skips=2, n_negative_samples=100, n_words=10000, vec_size=300, batch_size=16, window_size=5, learning_rate=0.2, n_epochs=15, do_plot=False):

        self.n_skips = n_skips
        self.n_negative_samples = n_negative_samples
        self.vec_size = vec_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_words = n_words

    def fit(self, documents):
        n_words_trained = 0
        tokens, vocab, data, frequencies, diction, reverse_diction = self._build_dataset(documents)
        n_tokens = len(tokens)
        n_vocab = len(vocab)
        words_per_epoch = n_vocab / self.n_epochs
        batch_logit, negative_samples_logit = self._build_graph(data,frequencies)
        loss = self._build_loss_metric(batch_logit, negative_samples_logit)
        optimizer = self._build_optimizer(loss, words_per_epoch, n_words_trained)
        self._train(data, optimizer, loss)



    def _build_dataset(self, documents):
        """Preprocesses the documents and creates the dataset for fitting."""

        # Create a specific tokenizer filtering out one length tokens
        additional_filters = [lambda token: len(token) == 1]
        token_filter = TokenFilter(filter_stop_words=False, additional_filters=additional_filters)
        preprocessor = Preprocessor(do_stem=False, do_lemmatize=False, parse_html=False, token_filter=token_filter, lower=False)

        # Get the term frequencies without idf
        tfidf = TfIdf(do_idf=False, preprocessor=preprocessor, n_words=self.n_words)
        tfidf.fit(documents)

        # Flatten the document tokens to make them on long list
        tokens = list(np.hstack(np.array(tfidf.document_tokens)))

        # Create the vocab list with 'UNK' for  vocab that couldn't make the vocab list
        vocab =  ['UNK'] + tfidf.vocab
        self.n_words += 1
        vocab_set = set(vocab)
        diction = {token: index for index, token in enumerate(vocab)}
        reverse_diction = dict(zip(diction.values(), diction.keys()))

        # Turn the long token list into a index references to the diction
        data = list(map(lambda token: diction[token] if token in vocab_set else 0, tokens))

        # Get the frequencies of tokens and add the frequency of 'UNK' at the beginning
        frequencies = np.insert(tfidf.total_term_freq, 0, data.count(0))[:self.n_words]

        return tokens, vocab, data, frequencies, diction, reverse_diction

    def _build_graph(self, data, frequencies):
        """Build the graph that is going to be trained."""

        self._batch_input = tf.placeholder(tf.int64, shape=[self.batch_size])
        self._labels_input = tf.placeholder(tf.int64, shape=[self.batch_size, 1])

        width = 0.5 / self.vec_size

        embeddings_dim = [self.n_words, self.vec_size]
        self._embeddings = tf.Variable(tf.random_uniform(embeddings_dim, minval=-width, maxval=width), name="embeddings")

        batch_embeddings = tf.nn.embedding_lookup(self._embeddings, self._batch_input)

        softmax_weights = tf.Variable(tf.zeros(embeddings_dim), name="softmax_weights")
        softmax_biases = tf.Variable(tf.zeros([self.n_words]), name="softmax_biases")

        # flattened_labels = tf.contrib.layers.flatten(self._labels_input)
        label_weights = tf.nn.embedding_lookup(softmax_weights, self._labels_input)
        label_biases = tf.nn.embedding_lookup(softmax_biases, self._labels_input)

        negative_sample_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(true_classes=self._labels_input, num_true=1, \
                                                                            num_sampled=self.n_negative_samples, unique=True, \
                                                                            range_max=self.n_words, distortion=0.75, unigrams=list(frequencies))

        negative_sample_weights = tf.nn.embedding_lookup(softmax_weights, negative_sample_ids)
        negative_sample_biases = tf.nn.embedding_lookup(softmax_biases, negative_sample_ids)

        batch_logit = tf.reduce_sum(tf.multiply(batch_embeddings, label_weights), 1) + label_biases

        reduced_negative_samples = tf.reshape(negative_sample_biases, [self.n_negative_samples])
        negative_samples_logit= tf.matmul(batch_embeddings, negative_sample_weights, transpose_b=True) + reduced_negative_samples

        return batch_logit, negative_samples_logit

    def _build_loss_metric(self, batch_logit, negative_samples_logit):
        """Build loss metric that will be optimized."""

        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(batch_logit), logits=batch_logit)
        negative_samples_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(negative_samples_logit), logits=negative_samples_logit)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        loss = (tf.reduce_sum(batch_loss) + tf.reduce_sum(negative_samples_loss)) / self.batch_size
        return loss

    def _build_optimizer(self, loss, words_per_epoch, n_words_trained):
        """Build the optimizer of the loss."""

        n_words_to_train = float(words_per_epoch * self.n_epochs)
        learning_rate = self.learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(n_words_trained, tf.float32) / n_words_to_train)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = optimizer.minimize(loss, global_step=tf.Variable(0, name="global_step"), \
                                    gate_gradients=optimizer.GATE_NONE)
        return optimizer

    def _train(self, data, optimizer, loss):
        start_index = 0
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)

            for epoch in range(self.n_epochs):
                avg_cost = 0
                n_batch = len(data) // self.batch_size
                for i in range(n_batch):
                    batch, labels, start_index = self._generate_batch(data, start_index)
                    _, error = sess.run([optimizer, loss], feed_dict = {self._batch_input: batch, self._labels_input: labels})

                    avg_cost += error / n_batch

                print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))

        print("\nTraining complete!")

    def _generate_batch(self, data, start_index):
        """Create a batch for a training step in Word2Vec."""

        # Initialize variables
        batch = np.zeros(self.batch_size)
        labels = np.zeros((self.batch_size, 1))
        span = 2 * self.window_size + 1
        buf = collections.deque(maxlen=span)

        # Get next data inside the span - wraps around when exceeds data length
        for _ in range(span):
            buf.append(data[start_index])
            start_index = (start_index + 1) % len(data)

        for i in range(self.batch_size // self.n_skips):
            target = self.window_size  # target label at the center of the buffer
            targets_to_avoid = [self.window_size]
            for j in range(self.n_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.n_skips + j] = buf[self.window_size]
                labels[i * self.n_skips + j, 0] = buf[target]
            buf.append(data[start_index])
            start_index = (start_index + 1) % len(data)
            # Backtrack a little bit to avoid skipping words in the end of a batch
            start_index = (start_index + len(data) - span) % len(data)

            flattened_labels = np.hstack(labels)
        return batch, labels, start_index
