""" Sub-network Handling """
from abc import ABC, abstractmethod
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser
from nncompress.backend.tensorflow_ import SimplePruningGate
from nncompress.backend.tensorflow_.transformation import parse, inject, cut, unfold
from nncompress.bespoke.base.generator import PruningGenerator

class ModelHouse(object):

    def __init__(self, model, custom_objects=None):

        model = unfold(model, custom_objects)
        self._model = model

        self._parser = PruningNNParser(model, custom_objects=custom_objects)
        self._parser.parse()

        self.custom_objects = custom_objects
        if self.custom_objects is None:
            self.custom_objects = {}
        self.custom_objects["SimplePruningGate"] = SimplePruningGate

        self.namespace = set() # For guaranteeing the uniqueness of additional layers.
        self.build()


    def build(self):
        min_layers = 5

        # construct t-rank
        v = self._parser.traverse()
        trank = {
            name:idx
            for idx, (name, _) in enumerate(v)
        }
        rtrank = {
            idx:name
            for name, idx in trank.items()
        }
        joints = set(self._parser.get_joints())

        groups = self._parser.get_sharing_groups()
        groups_ = []
        r_groups = {}
        for group in groups:
            group_ = []
            for layer_name in group:
                group_.append(self._model.get_layer(layer_name))
                r_groups[layer_name] = group_
            groups_.append(group_)

        def compute_constraints(layers):
            constraints = []
            for layer in layers:
                if layer.name in r_groups:
                    is_already = False
                    for c in constraints:
                        if c == r_groups[layer.name]:
                            is_already = True
                            break
                    if not is_already:
                        constraints.append(r_groups[layer.name])
            return constraints

        def is_compressible(layers):
            compressible = False
            for layer in layers:
                if layer.__class__.__name__ in ["Conv2D", "Dense", "DepthwiseConv2D"]:
                    compressible = True
                    break
            return compressible

        self._modules = []
        layers_ = []
        for idx in range(len(trank)):
            name = rtrank[idx]
            if len(layers_) > min_layers and name in joints:
                if is_compressible(layers_):
                    # make subnet from layers_                 
                    subnet = self._parser.get_subnet(layers_, self._model) 
                    constraints = compute_constraints(layers_)
                    module = ModuleHolder(subnet[0], subnet[1], subnet[2], constraints, self.namespace)
                    self._modules.append(module)
                layers_ = []
            else:
                layers_.append(self._model.get_layer(name))
        if len(layers_) > 0 and is_compressible(layers_):
            constraints = compute_constraints(layers_)
            subnet = self._parser.get_subnet(layers_, self._model)
            module = ModuleHolder(subnet[0], subnet[1], subnet[2], constraints, self.namespace)
            self._modules.append(module)

    def make_train_graph(self, scale=0.1, range_=None):
        """Make a training graph for this model house.

        """
        outputs = []
        outputs.extend(self._model.outputs)
        output_map = []
        for i, module in enumerate(self._modules):
            if range_ is not None and (i < range_[0] or i > range_[1]):
                continue

            for alter in module.alternatives:
                inputs_ = [ self._model.get_layer(layer).output for layer in module.inputs ]
                if type(module.subnet.inputs) == list:
                    out_ = alter(inputs_)
                else:
                    out_ = alter(inputs_[0])
                tout = [ self._model.get_layer(layer).output for layer in module.outputs ]
                if type(out_) != list:
                    tout = tout[0]
                    outputs.append(out_)
                else:
                    outputs.extend(out_)
                output_map.append((tout, out_))

        house = tf.keras.Model(self._model.inputs, outputs) # for test

        # add loss
        mse = tf.keras.losses.MeanSquaredError()
        for (t, s) in output_map:
            house.add_loss(mse(t, s)*scale)

        return house


    def extract(self, recipe):

        replacing_mappings = []
        in_maps = None
        ex_maps = []
        for idx, aidx in recipe.items():
            module = self._modules[idx]
            alters = module.get_alternative(aidx)
            
            is_change_shapes = module.is_change_shapes()
        
            for subnet, alter in alters:
                r, _, input_getter, output_ = alter
                if type(subnet) != list:
                    subnet = [subnet]

                # construct replacement arguments
                target = [ layer.name for layer in subnet ]
                replacement = json.loads(r.to_json())["config"]["layers"]
                ex_map = [
                    [(target[0], input_getter.name)],
                    [(target[-1], replacement[-1]["name"], 0, 0)]
                ]
                ex_maps.append(ex_map)
                replacing_mappings.append((target, replacement))

        print(replacing_mappings)

        # Conduct replace_block
        model_dict = self._parser.replace_block(replacing_mappings, in_maps, ex_maps, self.custom_objects)
        model_json = json.dumps(model_dict)
        ret = tf.keras.models.model_from_json(model_json, custom_objects=self.custom_objects)
        print(ret.summary())
        xxx

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

class ModuleHolder(object):
    """
        Can be a layer or layers.
        Abstract Layer

        Channel Pruning
        Identity
        Replace with the same input/output shape

    """
    
    def __init__(self, subnet, inputs, outputs, constraints, namespace, custom_objects=None):
        self.subnet = subnet
        self.inputs = inputs
        self.outputs = outputs
        self.constraints = constraints
        self.namespace = namespace
        self.alternatives = []
        self.custom_objects = custom_objects
        self.build()

    """
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
    """

    def build(self):
        pg = PruningGenerator(self.namespace)

        # Generate alternatives
        self.alternatives = pg.build(self.subnet)

class Alternative(object):

    def __init__(self, model):
        self._model = model


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
    house = mh.make_train_graph()

    tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)
    y = house(data)

    mh.query(0.5)


if __name__ == "__main__":
    test()
