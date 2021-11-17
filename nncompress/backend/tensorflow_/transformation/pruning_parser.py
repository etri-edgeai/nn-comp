from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import copy
from collections import OrderedDict

import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from orderedset import OrderedSet

from nncompress.backend.tensorflow_.transformation.handler import get_handler
from nncompress.backend.tensorflow_.transformation.parser import NNParser, serialize
from nncompress.backend.tensorflow_ import DifferentiableGate

class StopGradientLayer(tf.keras.layers.Layer):
    
    def __init__(self, name=None):
        super(StopGradientLayer, self).__init__(name=name)

    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)

    def get_config(self):
        return {
            "name":self.name
        }

class PruningNNParser(NNParser):
    """NNParser is a tool for enabling differentiable pruning.
   
    * Caution:
    Since it does not provide any additional loss to achieve target sparsity,
    you should define sparsity loss in somewhere not here.

    NNParser has a multi-di-graph defined in networkx.
    A node of a graph has two additional attributes: `layer_dict` and `nlevel`.
    `layer_dict` is a dictionary of the corresponding layer in the model dictionary, which can be
    converted to a JSON format.
    `nlevel` is the number of levels of a node (layer).
    If a layer is shared two times in a NN model, `nlevel` of it is 2.
    This feature is crucial to understand the working flow of a NN model.

    Similarly, an edge (src, dst) has three attributes: level_change, tensor, and inbound_idx.
    `level_change` is a tuple (x, y) where x is the level of src and y is that of dst.
    `tensor` is the tensor index of src.
    `inbound_idx` is the position of the edge in its flow of the inbound list of dst.

    Note that the inbound list of a node has multiple flows when it is shared multiple times.
    Thus, a flow has a separate inbounding edges, so that `inbound_idx` is a position over among edges.
    
    """

    def __init__(self, model, basestr="", custom_objects=None, gate_class=None):
        super(PruningNNParser, self).__init__(model, basestr=basestr, custom_objects=custom_objects)
        self._sharing_groups = []
        self._avoid_pruning = set()
        self._t2g = None
        if gate_class is None:
            self._gate_class = DifferentiableGate
        else:
            self._gate_class = gate_class

    def parse(self):
        super(PruningNNParser, self).parse()

        affecting_layers = self.get_affecting_layers()

        # Find sharing groups
        for layer, group  in affecting_layers.items():
            if len(group) == 0:
                continue
            group_ = OrderedSet([i[0] for i in group])
            candidates = []
            for target in self._sharing_groups:
                if len(group_.intersection(target)) > 0:
                    candidates.append(target)
            if len(candidates) == 0:
                self._sharing_groups.append(group_)
            elif len(candidates) == 1 and candidates[0] == group_: # to avoid kicking out the same group.
                continue
            else:
                new_group = OrderedSet()
                for cand in candidates:
                    new_group = new_group.union(cand)
                    self._sharing_groups.remove(cand)
                new_group = new_group.union(group_)
                self._sharing_groups.append(new_group)

        # Find layers to avoid pruning
        self._avoid_pruning = self.get_last_transformers()

    def get_affecting_layers(self):
        """This function computes the affecting layers of each node.

        """
        if self._model_dict is None:
            super(PruningNNParser, self).parse()

        affecting_layers = OrderedDict()
        for n in self._graph.nodes(data=True):
            for idx in range(n[1]["nlevel"]):
                affecting_layers[(n[0], idx)] = OrderedSet()

        def pass_(e):
            src = self._graph.nodes.data()[e[0]]
            dst = self._graph.nodes.data()[e[1]]
            level_change = e[2]["level_change"]
            if get_handler(src["layer_dict"]["class_name"]).is_transformer(e[2]["tensor"]):
                affecting_layers[(e[1], level_change[1])].add((e[0], level_change[0], e[2]["tensor"]))
            elif get_handler(dst["layer_dict"]["class_name"]).is_concat():
                affecting_layers[(e[1], level_change[1])] = OrderedSet() # Cancel previous info.
            else:
                if (e[0], level_change[0]) in affecting_layers: # To handle leaves
                    affecting_layers[(e[1], level_change[1])].update(affecting_layers[(e[0], level_change[0])])

        self.traverse(neighbor_callbacks=[pass_])
        return affecting_layers

    def get_last_transformers(self):
        """This function returns the names of last transformers whose units/channels are related to
            the network's output.

        # Returns.
            A set of last transformer names.
        """
        last = set()
        def stop_(e, is_edge):
            if is_edge:
                src = self._graph.nodes.data()[e[0]]
                if get_handler(src["layer_dict"]["class_name"]).is_transformer(e[2]["tensor"]):
                    last.add(e[0])
                    return True
            else:
                curr, level = e
                curr_data = self._graph.nodes.data()[curr]
                if get_handler(curr_data["layer_dict"]["class_name"]).is_transformer(0):
                    last.add(curr)
                    return True
            return False
        self.traverse(stopping_condition=stop_, inbound=True)
        return last

    def get_sharing_groups(self):
        """This function returns all the sharing groups in the networks.

        # Returns.
            a list of lists each of which is a sharing group.

        """
        ret = []
        for g in self._sharing_groups:
            ret.append(copy.deepcopy(g))
        return ret

    def get_sharing_layers(self, target):
        """This function returns the name of layers which are included in the same sharing group of `target`.

        # Arguments.
            target: a str, the name of a query layer.

        # Returns.
            a list of str, the name of sharing group layers.

        """
        target_group = None
        for g in self._sharing_groups:
            if target in g:
                target_group = g
        if target_group is None:
            raise ValueError("No sharing group for %s" % target)
        return [ copy.deepcopy(i) for i in target_group ]
 
    def _reroute(self, at, target, layers_dict):
        """
            Assumption:
                - `at` is already connected into `target`.
        """
        node_data, level, tensor = at
        for e in self._graph.out_edges(node_data["config"]["name"], data=True):
            src, dst, level_change, tensor_ = e[0], e[1], e[2]["level_change"], e[2]["tensor"]
            # TODO: Better implementation
            if level != level_change[0] or tensor != tensor_:
                continue
            for flow_idx, flow in enumerate(layers_dict[dst]["inbound_nodes"]):
                for inbound in flow:
                    if inbound[0] == src and level_change[0] == inbound[1] and level_change[1] == flow_idx and tensor_ == inbound[2]:
                        inbound[0] = target[0]["config"]["name"]
                        inbound[1] = target[1]
                        inbound[2] = target[2]

    def get_first_activation(self, node_name):
        
        act = []
        def act_mapping(n, level):
            node_data = self._graph.nodes(data=True)[n]
            if node_data["layer_dict"]["class_name"] == "Activation" and len(act) == 0:
                act.append(node_data["layer_dict"]["config"]["name"])
        
        def stop(n, is_edge=False):
            if len(n) > 2:
                return False

            n, level = n
            if len(act) > 0:
                return True
            else:
                return False

        sources = [ (node_name, self._graph.nodes(data=True)[node_name]) ]
        self.traverse(sources=sources, node_callbacks=[act_mapping], stopping_condition=stop)

        return act[0]


    def inject(self, avoid=None, with_mapping=False, with_splits=False):
        """This function injects differentiable gates into the model.

        # Arguments.
            avoid: a set or a list, the layer names to avoid.

        # Returns.
            a Keras model, which has differentiable gates.

        """
        self._t2g = {}

        if avoid is None:
            avoid = set()
        if type(avoid) != set: #setify
            avoid = set(avoid)
        avoid = avoid.union(self._avoid_pruning)

        model_dict = copy.deepcopy(self._model_dict)
        layers_dict = {}
        for layer_dict in model_dict["config"]["layers"]:
            layers_dict[layer_dict["config"]["name"]] = layer_dict

        if with_splits:
            sharing_groups_ = []
            for g in self._sharing_groups:
                sharing_groups_ += [
                    [target]
                    for target in g
                ]
        else:
            sharing_groups_ = self._sharing_groups

        gate_mapping = {}
        for group in sharing_groups_:
            if len(avoid.intersection(group)) > 0:
                continue
            
            # Create a gate
            channels = self.get_nchannel(list(group)[0])
            gate = self._gate_class(channels, name=self.get_id("gate"))
            gate_dict = serialize(gate)
            model_dict["config"]["layers"].append(gate_dict)

            for target in group:
                self._t2g[target] = gate.name
                n = self._graph.nodes[target] 
                # add n into gate's inbound_nodes
                nflow = max(n["nlevel"], 1)
                for i in range(nflow):
                    gate_dict["inbound_nodes"].append([[target, i, 0, {}]])
                    gate_level = len(gate_dict["inbound_nodes"])-1
                    self._reroute(at=(n["layer_dict"], i, 0), target=(gate_dict, gate_level, 0), layers_dict=layers_dict)
                    gate_mapping[(target, i)] = gate_dict, gate_level

        # Used in traversing
        def modify_output(n, level):
            node_data = self._graph.nodes[n]
            h = get_handler(node_data["layer_dict"]["class_name"])
            if h.is_transformer(0): 
                return

            if (n, level) not in gate_mapping:
                # update gate_mapping
                gates = []
                for e in self._graph.in_edges(n, data=True):
                    src, dst, level_change, tensor = e[0], e[1], e[2]["level_change"], e[2]["tensor"]
                    if level_change[1] != level:
                        continue
                    if (src, level_change[0]) in gate_mapping:
                        gates.append(gate_mapping[(src, level_change[0])])
                    else:
                        gates.append(None)

                if len(gates) == 0:
                    return
                continue_ = False
                for gate in gates:
                    if gate is not None:
                        continue_ = True
                if not continue_:
                    return

                gmodifier = h.get_gate_modifier(self.get_id("gate_modifier"))
                if gmodifier is None:
                    gate_dict, gate_level = gates[0]
                    self.restore_id("gate_modifier")
                else:
                    gate_dict = serialize(gmodifier)
                    gate_level = 0
                    model_dict["config"]["layers"].append(gate_dict)
                    inbound = []
                    for gate in gates:
                        if gate is not None:
                            gate_dict_, gate_level_ = gate
                        else: # Handling gate is None (no pruning)
                            channel = self.get_nchannel(src)
                            lambda_dict = serialize(Lambda(lambda x: tf.ones_like(tf.math.reduce_sum(x, axis=-1)), name=self.get_id("ones")))
                            lambda_dict["inbound_nodes"].append([
                                [src, level_change[0], tensor, {}]
                            ])
                            model_dict["config"]["layers"].append(lambda_dict)
                            gate_dict_ = lambda_dict
                            gate_level_ = 0
                        tensor = 1 if gate_dict_["class_name"] == self._gate_class.__name__ else 0
                        inbound.append([gate_dict_["name"], gate_level_, tensor, {}])
                    gate_dict["inbound_nodes"].append(inbound)
                gate_mapping[(n, level)] = gate_dict, gate_level

            # modify ouptut upon its modifier
            gate_dict, gate_level = gate_mapping[(n, level)]
            modifier = h.get_output_modifier(self.get_id("output_modifier"))
            if modifier is None:
                self.restore_id("output_modifier")
                return
            modifier_dict = serialize(modifier)
            model_dict["config"]["layers"].append(modifier_dict)

            #stop_gradient = Lambda(lambda x: tf.stop_gradient(x), name=self.get_id("stop_gradient"))
            stop_gradient = StopGradientLayer(name=self.get_id("stop_gradient"))
            stop_gradient_dict = serialize(stop_gradient)
            model_dict["config"]["layers"].append(stop_gradient_dict)

            # Make connections
            tensor = 1 if gate_dict["class_name"] == self._gate_class.__name__ else 0
            stop_gradient_dict["inbound_nodes"].append([[gate_dict["name"], gate_level, tensor, {}]])
            modifier_dict["inbound_nodes"].append([[n, level, 0, {}], [stop_gradient.name, 0, 0, {}]])
            self._reroute(at=(node_data["layer_dict"], level, 0), target=(modifier_dict, 0, 0), layers_dict=layers_dict)

        # create sub-layers to handle shift op.
        self.traverse(node_callbacks=[modify_output])

        model_json = json.dumps(model_dict)
        custom_objects = {self._gate_class.__name__:self._gate_class, "StopGradientLayer":StopGradientLayer}
        custom_objects.update(self._custom_objects)
        ret = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
        for layer in self._model.layers:
            ret.get_layer(layer.name).set_weights(layer.get_weights())

        if with_mapping:
            return ret, gate_mapping
        else:
            return ret

    def get_t2g(self):
        """Returns the mapping from targets to gates.

        # Returns
            A dictionary from targets to gates.

        """
        return copy.deepcopy(self._t2g)

    def clear(self):
        """Clear internal info.
        It does not remove the parsed information.

        """
        self._t2g = None
        self._id_cnt = {}

    def cut(self, gmodel, return_history=False):
        """This function gets a compressed model from a model having gates

        # Arguments.
            gmodel: a Keras model, which has differentiable gates.

        # Returns.
            a Keras model, which is compressed.

        """
        model_dict = copy.deepcopy(self._model_dict)
        layers_dict = {}
        for layer_dict in model_dict["config"]["layers"]:
            layers_dict[layer_dict["config"]["name"]] = layer_dict

        gmodel_dict = json.loads(gmodel.to_json())
        glayers_dict = {}
        g2t = {}
        gate_mapping = {}
        for layer_dict in gmodel_dict["config"]["layers"]:
            glayers_dict[layer_dict["config"]["name"]] = layer_dict
            if layer_dict["class_name"] == self._gate_class.__name__:
                g2t[layer_dict["name"]] = set()
                gate = gmodel.get_layer(layer_dict["name"]).binary_selection() == 1.0
                for flow in layer_dict["inbound_nodes"]:
                    for inbound in flow:
                        g2t[layer_dict["name"]].add(inbound[0])
                        if inbound[0] not in gate_mapping:
                            gate_mapping[(inbound[0], inbound[1])] = gate

        history = {n:None for n in self._graph.nodes}
        weights = {
            n:gmodel.get_layer(n).get_weights() for n in self._graph.nodes
            if gmodel.get_layer(n).__class__.__name__ != "Functional"
        }

        def cut_weights(n, level):

            node_data = self._graph.nodes[n]
            h = get_handler(node_data["layer_dict"]["class_name"])

            # update gate_mapping
            gates = [] # input_gates
            for e in self._graph.in_edges(n, data=True):
                src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]

                if level_change[1] != level:
                    continue
                elif (src, level_change[0]) in gate_mapping:
                    gates.append(gate_mapping[(src, level_change[0])])
                elif glayers_dict[n]["class_name"] != "Functional":
                    # TODO: better implementation for getting the channel of src?
                    gates.append(np.ones((self.get_nchannel(src),)) == 1.0) # Holder

            if len(gates) == 0:
                return

            input_gate = h.update_gate(gates, self._model.get_layer(n).input_shape)
            if input_gate is None:
                input_gate = gates[0]
            if (n, level) not in gate_mapping and not h.is_transformer(0):
                gate_mapping[(n, level)] = input_gate # gate transfer

            output_gate = gate_mapping[(n, level)] if (n, level) in gate_mapping else None
            if not history[n]:
                new_weights = h.cut_weights(weights[n], input_gate, output_gate)
                weights[n] = new_weights
                h.update_layer_schema(layers_dict[n], weights[n], input_gate, output_gate)
                history[n] = (input_gate, output_gate)

        self.traverse(node_callbacks=[cut_weights])

        model_json = json.dumps(model_dict)
        ret = tf.keras.models.model_from_json(model_json, custom_objects=self._custom_objects)
        for layer in ret.layers:
            if layer.name in weights:
                layer.set_weights(weights[layer.name])
            else:
                print(layer.name, " is not in `weights`. It should be handled somewhere.")

        if return_history:
            ret = (ret, history)
        return ret

if __name__ == "__main__":

    from tensorflow import keras
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    #model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
    model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)

    tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

    parser = PruningNNParser(model)
    parser.parse()
    model_ = parser.inject()
    tf.keras.utils.plot_model(model_, to_file="gmodel.png", show_shapes=True)

    cmodel = parser.cut(model_)

    tf.keras.utils.plot_model(cmodel, to_file="cmodel.png", show_shapes=True)

    # random data test
    data = np.random.rand(1,32,32,3)
    y1 = model_(data)
    y2 = cmodel(data)

    print(y1)
    print(y2)
