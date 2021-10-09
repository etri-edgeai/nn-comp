import tensorflow as tf
from tensorflow import keras


class WeightedSum(keras.layers.Layer):

    def __init__(self, n, initializer=tf.zeros_initializer()):

        self.w = tf.Variable(
            initial_value=initializer(shape=(n,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        output_ = tf.stack(inputs, axis=1)
        return tf.math.reduce_sum(output_ * self.w, axis=1)

def make_nn_from_graph(g):

    for n in g.nodes:

        

