import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, ReLU, Activation


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

    def __call__(self, s1, s2):
        e1 = self.embedding(s1)
        e1 = tf.math.reduce_mean(e1, 0)
        e2 = self.embedding(s2)
        e2 = tf.math.reduce_mean(e2, 0)

        sub = tf.math.abs(e2 - e1)
        prod = e1 * e2

        return self.model(tf.stack((sub, prod), 1))

    def encode(self, label):
        min_l, max_l = self.label_range

        y = label - min_l
        y /= max_l - min_l
