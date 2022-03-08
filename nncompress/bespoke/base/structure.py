""" Sub-network Handling """
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold

class ModelHouse(object):

    def __init__(self, model, custom_objects=None):

        model = unfold(model, custom_objects)

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
            i: PruningModuleHolder(group)
            for i, group in enumerate(groups_)
        }

        outputs = []
        outputs.extend(self._model.outputs)
        for _, module in self._modules.items():
            for name, alters in module.alternative.items():
                for alter, out_ in alters:
                    outputs.append(out_)

        self.inputs = self._model.inputs
        self.outputs = outputs

        #self.house = tf.keras.Model(self._model.input, outputs) # for test
        #self.self_distillation_loss()

    def add_self_distillation_loss(self, house, scale=0.1):
        # Compute self-distillation-loss
        mse = tf.keras.losses.MeanSquaredError()
        for i, module in self._modules.items():
            for name, alters in module.alternative.items():
                # diff
                layer = house.get_layer(name)       
                for alter, out_ in alters:
                    house.add_loss(mse(layer.output, out_)*scale)

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
                for layer in module.get_subnet_as_list():
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
            alters = module.alternative[aid]

            # use parser to replace module with alters
            # ...


class ModuleHolder(ABC):
    """
        Can be a layer or layers.
        Abstract Layer

        Channel Pruning
        Identity
        Replace with the same input/output shape

    """
    
    def __init__(self, subnet, custom_objects=None):
        self.subnet = subnet
        self.alternative = None
        self.build()

    @property
    def subnet(self):
        if len(self._subnet) == 1:
            return self._subnet[0]
        else:   
            return self._subnet

    @subnet.setter
    def subnet(self, subnet_):
        if type(subnet_) == list:
            self._subnet = subnet_
        else:
            self._subnet = [subnet_]

    def get_subnet_as_list(self):
        return self._subnet

    @abstractmethod
    def build(self):
        """Build the index for this module

        """

    def compute_params(self, idx):
        if idx == -1:
            layer_params = 0
            for layer in self.get_subnet_as_list():
                for w in layer.get_weights(): 
                    layer_params += np.prod(w.shape)
            return layer_params
        else:
            layer_params = 0
            for name, alters in self.alternative.items():
                alter, out_ = alters[idx]
                for layer in alter.layers:
                    for w in layer.get_weights():
                        layer_params += np.prod(w.shape)
            return layer_params

class PruningModuleHolder(ModuleHolder):

    def __init__(self, subnet, custom_objects=None):
        self.scales = [0.25, 0.5, 0.75]
        super(PruningModuleHolder, self).__init__(subnet, custom_objects)

    def compute_params(self, idx):
        if idx == -1:
            return super(PruningModuleHolder, self).compute_parmas(idx)
        else:
            sum_ = 0
            for name, alters in self.alternatives.items():
                alter, out_ = alters[idx]
                pruned = alter.layers[0]
                gate = alter.layers[1]
                pw = pruned.get_weights()[0][:,:,:,gate.gates == 1.0]
                print(pruned.get_weights()[0].shape)
                print(pw.shape)
                print(np.count_nonzero(gate.gates))

    def build(self):
        self.alternative = {} # init alternative

        for idx, scale in enumerate(self.scales):

            w = self._subnet[0].get_weights()[0] # we can use another criterion for computing mask.
            w = np.abs(w)
            sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
            sorted_ = np.sort(sum_, axis=None)
            val = sorted_[int((len(sorted_)-1)*scale)]
            keep = (sum_ >= val).astype(np.float32)

            layers = self._subnet
            for layer in layers:
                if layer.name not in self.alternative:
                    self.alternative[layer.name] = []

                # Assumption: convolution
                config = layer.get_config()
                config["name"] = config["name"] + "_prune_" + str(idx)
                pruned = tf.keras.layers.Conv2D.from_config(config)
                gate = SimplePruningGate(config["filters"])
                gate.collecting = False

                input_ = tf.keras.layers.Input(layer.input_shape[1:])
                gate(pruned(input_))
                gate.gates.assign(keep)
                model_ = tf.keras.Model(inputs=input_, outputs=gate.output)

                output = model_(layer.input)
                self.alternative[layer.name].append((model_, output[0])) # gate returns two tensors.




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

    tf.keras.utils.plot_model(mh.house, to_file="house.pdf", show_shapes=True)

    # random data test
    data = np.random.rand(1,32,32,3)
    y = mh.house(data)

    mh.query(0.5)


if __name__ == "__main__":
    test()
