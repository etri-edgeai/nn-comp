""" Sub-network Handling """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser

class ModelHouse(object):

    def __init__(self, model, custom_objects=None):
       
        self._parser = PruningNNParser(model, custom_objects=custom_objects)
        self._parser.parse()

        groups = self._parser.get_sharing_groups()
        groups_ = []
        for group in groups:
            group_ = []
            for layer_name in group:
                group_.append(model.get_layer(layer_name))
            groups_.append(group_)

        self._model = model
        self._modules = {
            i: ModuleHolder(i, group)
            for i, group in enumerate(groups_)
        }

        outputs = []
        outputs.extend(self._model.outputs)
        for i, module in self._modules.items():
            for name, alters in module.alternatives.items():
                for alter in alters:
                    outputs.append(alter[-1].output) # projection

        self.house = tf.keras.Model(self._model.input, outputs) # for test
        self.self_distillation_loss()

    def self_distillation_loss(self):
        # Compute self-distillation-loss
        for i, module in self._modules.items():
            for name, alters in module.alternatives.items():
                # diff
                layer = self.house.get_layer(name)       
                for alter in alters:
                    mse = tf.keras.losses.MeanSquaredError()
                    self.house.add_loss(mse(layer.output, alter[-1].output))

    def query(self, ratio):
        """Lightweight model with compression ratio.

        Baseline

        """
        params = self._model.count_params()
        new_params = params
        compressed = {}
        while params * ratio < new_params:

            # find max param layer
            p = 0
            p_i = -1
            for i, module in self._modules.items():
                layer_params = 0
                for layer in module.layers:
                    layer_params += np.prod(layer.get_weights()[0].shape)
                if i not in compressed:
                    compressed[i] = -1

                if len(module.scales) - 1 == compressed[i]:
                    continue

                gain = layer_params - layer_params * module.scales[compressed[i]+1]
                if gain > p:
                    p = gain
                    p_i = i

            compressed[p_i] += 1
            new_params -= p

        for i, aid in compressed.items():
            module = self._modules[i]
            alters = module.alternatives[aid]

            # use parser to replace module with alters
            # ...


class ModuleHolder(object):
    """
        Can be a layer or layers.

    """
    
    def __init__(self, idx, group, custom_objects=None):
        self.layers = [ l for l in group ]
        self.alternatives = None
        self.scales = [0.5, 0.75, 0.875]

        self.build()

    def build(self):
        self.alternatives = {}

        for idx, scale in enumerate(self.scales):

            # setup common mask
            w = self.layers[0].get_weights()[0] # we can use another criterion for computing mask.
            w = np.abs(w)
            sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
            sorted_ = np.sort(sum_, axis=None)
            val = sorted_[int((len(sorted_)-1)*scale)]
            removal = (sum_ >= val).astype(np.float32)

            for layer in self.layers:
                if layer.name not in self.alternatives:
                    self.alternatives[layer.name] = []

                # Assumption: convolution
                config = layer.get_config()
                old = config["filters"]
                config["filters"] = config["filters"] - np.sum(removal)
                config["name"] = config["name"] + "_" + str(idx)

                pruned = layers.Conv2D.from_config(config)
                expand = layers.Conv2D(old, (1,1), name=config["name"]+"_projection")
                self.alternatives[layer.name].append([pruned, expand])

                # draw model
                pruned(layer.input)
                expand(pruned.output)

def test():

    """
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
    """

    from tensorflow import keras

    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
    #model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)

    tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

    mh = ModelHouse(model)

    # random data test
    data = np.random.rand(1,32,32,3)
    y = mh.house(data)

    mh.query(0.5)


if __name__ == "__main__":
    test()
