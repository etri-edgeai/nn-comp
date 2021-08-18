""" Sub-network Handling """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser

class Network(object):

    def __init__(self, model, custom_objects=None):
       
        #self._parser = PruningNNParser(model, custom_objects=custom_objects)
        #self._parser.parse()

        self._model = model
        self._ops = {
            layer.name:None
            for layer in model.get_layers()
        }


class SubNetwork(object):
    """
        Can be a layer or a model

    """
    
    def __init__(self, block, custom_objects=None):
        pass


def test():

    class MyModel(tf.keras.Model):
      def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(10, activation='softmax')

      def call(self, x):
        x = self.d1(x)
        return self.d2(x)


    model = MyModel()

    inputs = layers.Input(shape=(100,))
    x = layers.Dense(512, activation=tf.nn.gelu)(inputs)
    x = layers.Dense(512, activation=tf.nn.gelu)(x)
    #x = model(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    inputs = layers.Input(shape=(100,))
    x = model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=x)

    print(model.to_json())

    print(model.summary())

if __name__ == "__main__":
    test()
