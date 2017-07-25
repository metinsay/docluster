import tensorflow as tf

class Word2Vec(object):

    def __init__(self, vec_size=300, batch_size=16, window_size=5, learning_rate=0.2, epochs=15, do_plot=False):

        self.vec_size = vec_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.word_to_id_map = {}

        self.build_graph()

    def build_graph(self):


    def train(self, data):



        embeddings_size = [vocab_size, self.vec_size]
        width = 0.5 / self.vec_size
        embeddings = tf.random_uniform(embeddings_size, minval=-width, maxval=width)

        softmax_weights = tf.Variable(tf.zeros(embeddings_size), name='softmax_weights')
        softmax_bias = tf.Variable(tf.zeros([vocab_size]), name='softmax_bias')
        glob_step = tf.Variable(0, name='glob_step')
