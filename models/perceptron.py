import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, ReLU, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence


class Perceptron(tf.Module):
    def __init__(self, vocab, emb_dim, output_sizes, label_range):
        super().__init__(name='perceptron')

        self.embedding = Embedding(len(vocab), emb_dim)
        self.model = tf.keras.Sequential()
        input_size = 2 * emb_dim
        with self.name_scope:
            for size in output_sizes[:-1]:
                self.model.add(Dense(input_dim=input_size, output_size=size, activation='relu'))
                input_size = size
            self.model.add(Dense(input_dim=input_size, output_size=output_sizes[-1], activation='log_softmax'))

        self.label_range = label_range

    def data_embedding(self, s1, s2):
        e1 = self.embedding(s1)
        e1 = tf.math.reduce_mean(e1, 0)
        e2 = self.embedding(s2)
        e2 = tf.math.reduce_mean(e2, 0)

        sub = tf.math.abs(e2 - e1)
        prod = e1 * e2

        return tf.stack((sub, prod), 1)

    def encode_label(self, label):
        min_l, max_l = self.label_range

        y = label - min_l
        y /= max_l - min_l

    def __call__(self, s1, s2):
        return self.model(self.data_embedding(s1, s2))

class Neural(Perceptron):
    def __init__(self, vocab, output_sizes, emb_dim=300, label_range=(0,5), optimizer='adam', loss='kl_divergence'):
        super().__init__(vocab, emb_dim, output_sizes, label_range)

        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, train_data, labels, epochs=5):
        X, Y = list(map(self.data_embedding, train_data)), list(map(self.encode_label, labels))

        self.model.fit(X, Y, epochs=epochs, use_multiprocessing=True)