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

def serialize(layer):
    layer_dict = tf.keras.layers.serialize(layer)
    layer_dict["name"] = layer.name
    layer_dict["inbound_nodes"] = []
    return layer_dict

class NNParser(object):
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

    def __init__(self, model, basestr="", custom_objects=None, namespace=None):

        if type(model) == keras.Sequential:
            input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape, name="seq_input")
            prev_layer = input_layer
            for layer in model.layers:
                layer._inbound_nodes = []
                prev_layer = layer(prev_layer)
            model = keras.models.Model([input_layer], [prev_layer])

        self._model = model
        self._custom_objects = custom_objects or {}

        self._graph = nx.MultiDiGraph()
        self._model_dict = None

        self._id_cnt = {}
        self._basestr = basestr
        if namespace is None:
            self._namespace = set()
        else:
            self._namespace = namespace

    def copy_model(self):
        model = tf.keras.models.clone_model(self._model)
        model.set_weights(self._model.get_weights())
        return model

    @property
    def model(self):
        return self._model

    @property
    def custom_objects(self):
        return self._custom_objects

    def get_id(self, prefix):
        """This function gives an identifier for a prefix.

        # Arguments.
            prefix: str, a prefix of identifiers (scope).
        # Returns.
            a str, the identifier under the given prefix scope.

        """
        ret = None
        while ret is None:
            if prefix not in self._id_cnt:
                self._id_cnt[prefix] = 0
            else:
                self._id_cnt[prefix] += 1

            if self._id_cnt[prefix] == 0:
                ret = self._basestr + prefix
            else:
                ret = self._basestr + prefix + "_" + str(self._id_cnt[prefix])
            if self._namespace is not None and ret in self._namespace:
                ret = None # duplication
        self._namespace.add(ret)
        return ret

    def get_nodes(self, ids):
        """Return the nodes upon `ids`

        # Arguments.
            ids: a list of str, the ids to retrieve

        # Returns.
            a list of the node instances in the graph

        """
        return [(id_, self._graph.nodes[id_]) for id_ in ids]

    def get_layer_dict(self, name):
        for layer_dict in self._model_dict["config"]["layers"]:
            if layer_dict["config"]["name"] == name:
                return copy.deepcopy(layer_dict)
        return None

    def restore_id(self, prefix):
        """This function is used to restore the id counter of `get_id`.
        Suppose that you claim some identifier and it is turned out to be useless.
        In this case, there are two options: ignoring it or revoking it.
        This function is given for the latter.

        # Arguments:
            prefix: str, a prefix of identifiers (scope).

        """
        assert prefix in self._id_cnt
        assert self._id_cnt[prefix] >= 0
        self._id_cnt[prefix] -= 1

    def get_nchannel(self, name, inbound=False):
        """Get the number of output/input channels of the layer defined by `name`.
        The layer should be found in the input model.

        # Arguments.
            name: str, a layer name
        # Returns.
            int, The number of output channels.

        """
        # TODO: handling multiple in/out shapes
        layer = self._model.get_layer(name)
        if inbound:
            input_shape = layer.get_input_shape_at(0)
            if type(input_shape) == tuple:
                return input_shape[-1]
            elif type(input_shape) == list:
                return input_shape[0][-1]
            else:
                raise NotImplementedError("`output_shape` is neither a list nor a tuple.")
        else:
            output_shape = layer.get_output_shape_at(0)
            if type(output_shape) == tuple:
                return output_shape[-1]
            elif type(output_shape) == list:
                return output_shape[0][-1]
            else:
                raise NotImplementedError("`output_shape` is neither a list nor a tuple.")

    def replace_block(self, replace_mappings, in_maps=None, ex_maps=None, custom_objects=None):
        """This function replaces a block with another block in your NN model.
        A block is defined to be a list of layers each of which is written in a dictionary.
        
        # Arguments.
            replace_mappings: list of tuples (replaced, replacement), each element is a list of layers.
            in_maps: None, str, or list, if it is a list, the length of it is the same as that of `replace_mappings`.
                Each element is 'seq' or None. If an element is 'seq', then the corresponding replacement is
                supposed to be connected sequentially. Otherwise, the connections between layers in the replacement
                are already expressed in the `inbound nodes` attributes.
                If `in_maps` is not list, then it will be broadcasted to the same value over replace_mappings.
            ex_maps: None, a list, an element is a list having two elements. The first one is a mapping from
                layers to have external inputs in the target block to those in the replacement block.
                Similarly, the second one is a mapping from the layers to have external outputs in the target block
                to those in the replacement block.
                Note that the first element has a form of (target_layer_name, replacement_layer_name),
                while the second element has a form of (target_layer_name, replacement_layer_name, target_tensor, replacement_tensor).
                `target_tensor` is the tensor index of the target used in an inbound node list.
                `replacemenet_tensor` is similarly defined.
            custom_objects: `custom objects for loading a Keras model having custom layers/operators.
        # Returns.
            new_model_dict
        """
        if self._model_dict is None:
            raise ValueError("`parse` should be executed before doing `replace_block`.")

        if type(in_maps) != list:
            in_maps = [in_maps for _ in range(len(replace_mappings))] # repeat
        if type(ex_maps) != list:
            ex_maps = [ex_maps for _ in range(len(replace_mappings))] # repeat

        model_dict = copy.deepcopy(self._model_dict)
        layers_dict = {}
        for layer_dict in model_dict["config"]["layers"]:
            layers_dict[layer_dict["config"]["name"]] = layer_dict

        map_ = {}
        for (target, replacement), in_map, ex_map in zip(replace_mappings, in_maps, ex_maps):
            new_items = set([r["name"] for r in replacement])
            n2i = { r["name"]:i for i, r in enumerate(replacement) }
            if type(target) != list:
                target = [target]

            if ex_map is None:
                ex_map = [[(target[0], replacement[0]["name"])], [(target[-1], replacement[-1]["name"], 0, 0)]]

            # Extract block flows
            target_set = set(target)
            gates = {t:[] for t in target}
            inputs = set()
            for t in target:
                node_data = self._graph.nodes.data()[t]
                for flow_idx, flow in enumerate(node_data["layer_dict"]["inbound_nodes"]):
                    for inbound in flow:
                        if inbound[0] not in inputs and inbound[0] not in target_set: # outer
                            inputs.add(inbound[0])
                            if flow_idx not in gates[t]:
                                gates[t].append(flow_idx) # main flow from external sources.

            offsets = { r["name"]:len(r["inbound_nodes"]) for r in replacement }
            has_inbound = {r:len(replacement[n2i[r]]["inbound_nodes"]) > 0 for r in n2i}
            for flow_idx in range(len(gates[target[0]])):
                # Handling inputs to replacement block.
                for t, r in ex_map[0]:
                    flow = self._graph.nodes.data()[t]["layer_dict"]["inbound_nodes"][gates[t][flow_idx]]
                    if has_inbound[r]:
                        # The first flow is the input from out of block.
                        if flow_idx == 0:
                            backup = copy.deepcopy(replacement[n2i[r]]["inbound_nodes"])
                            replacement[n2i[r]]["inbound_nodes"] = [copy.deepcopy(flow)]
                        else:
                            replacement[n2i[r]]["inbound_nodes"].append(copy.deepcopy(flow))
                            for r_idx, r_flow in enumerate(backup[1:]):
                                r_flow = copy.deepcopy(r_flow)
                                for inbound in r_flow:
                                    inbound[1] = inbound[1] + offsets[inbound[0]] * flow_idx
                                replacement[n2i[r]]["inbound_nodes"].append(r_flow)
                    else:
                        replacement[n2i[r]]["inbound_nodes"].append(copy.deepcopy(flow))

                if in_map == "seq":
                    offsets = { r["name"]:1 for r in replacement }
                    for r_idx, r in enumerate(replacement[1:]):
                        r["inbound_nodes"].append([[replacement[r_idx]["name"], len(replacement[r_idx]["inbound_nodes"])-1, 0, {}]])
                else:
                    # The first layer does not have any inbound.
                    if flow_idx == 0: # backup flow info.
                        backup = []
                        for r in replacement[1:]:
                            backup.append(copy.deepcopy(r["inbound_nodes"]))
                    else:
                        for r_idx, r in enumerate(replacement[1:]):
                            r_flow = copy.deepcopy(backup[r_idx])
                            for inbound in r_flow:
                                inbound[1] = inbound[1] + offsets[inbound[0]] * flow_idx
                            r["inbound_nodes"].append(r_flow)

                for t, r, t_tensor, r_tensor in ex_map[1]:
                    map_ [t] = r
                    r_flow_idx = offsets[r] * (flow_idx + 1) - 1
                    nflow = len(self._graph.nodes.data()[t]["layer_dict"]["inbound_nodes"])
                    assert nflow % len(gates[target[0]]) == 0
                    t_flow_idx = (nflow / len(gates[target[0]])) * (flow_idx + 1) - 1

                    # Handling outputs
                    for e in self._graph.out_edges(t, data=True):
                        src, dst, level_change, tensor = e[0], e[1], e[2]["level_change"], e[2]["tensor"]
                        for flow_ in layers_dict[dst]["inbound_nodes"]:
                            for inbound in flow_:
                                if inbound[0] == src and inbound[1] == level_change[0] and inbound[1] == t_flow_idx and inbound[2] == tensor and inbound[2] == t_tensor:
                                    inbound[0] = r
                                    inbound[1] = r_flow_idx
                                    inbound[2] = r_tensor
                                    # level and tensor idx are not changed.

            to_remove = []
            for layer_dict in model_dict["config"]["layers"]:
                if layer_dict["name"] in target_set:
                    to_remove.append(layer_dict)
            for r in to_remove:
                model_dict["config"]["layers"].remove(r)
            model_dict["config"]["layers"].extend(replacement)

        # hard remedy
        for layer in model_dict["config"]["layers"]:
            for flow in layer["inbound_nodes"]:
                for inbound in flow:
                    if inbound[0] in map_:
                        inbound[0] = map_[inbound[0]]


        return model_dict

    def get_randomwalk(self, start, p=0.25, types=None, min_step=-1):

        trail = []
        def callback_(n, level):
            node_data = self._graph.nodes[n]
            name = node_data["layer_dict"]["config"]["name"]
            trail.append(name)

        def stop_cond(e, is_edge):
            if is_edge:
                return False
            if min_step != -1 and len(trail) >= min_step:
                return True
            curr, level = e
            if types is not None:
                if self._model.get_layer(curr).__class__ in types:
                    if np.random.random() > p:
                        return True
                    else:
                        return False
            return False

        s = (start, self._graph.nodes(data=True)[start])
        self.traverse(node_callbacks=[callback_], sources=[s], stopping_condition=stop_cond)
        return trail


    def get_joints(self, filter_=None, start=None, min_step=-1):

        if len(self.torder) != self._graph.number_of_nodes():
            return

        max_idx = [-1]
        joints = []
        # Finding candidate joints
        def callback_(n, level): 
            node_data = self._graph.nodes[n]
            name = node_data["layer_dict"]["config"]["name"]
            if max_idx[0] <= self.torder[name]:
                if filter_ is not None:
                    if filter_(node_data):
                        joints.append(name)
                else:
                    joints.append(name)

            neighbors = self._graph.out_edges(name, data=True)
            for e in neighbors:
                src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
                if max_idx[0] < self.torder[dst] and (start is None or (start is not None and src != start)):
                    max_idx[0] = self.torder[dst]

        def stop_cond(e, is_edge):
            if is_edge:
                return False
            if min_step != -1 and len(trail) >= min_step:
                return True
            return False

        if start is not None:
            s = (start, self._graph.nodes(data=True)[start])
            self.traverse(node_callbacks=[callback_], sources=[s], stopping_condition=stop_cond)
        else:
            self.traverse(node_callbacks=[callback_], stopping_condition=stop_cond)
        return joints

    def first_common_descendant(self, names, joints, is_transforming=True):

        joints = set(joints)
        sources = [ x for x in self._graph.nodes(data=True) if x[1]["layer_dict"]["config"]["name"] in names]
        visits = []
        for s in sources:
            v = self.traverse(sources=[s], sync=False)
            v = [ (v_[0], self.torder[v_[0]]) for v_ in v ]
            v = sorted(v, key=lambda x: -1*x[1])
            v.pop() # remove self
            visits.append(v)

        no_match = False
        while True:
            for v in visits:
                if len(v) == 0:
                    no_match = True
                    break
            if no_match:
                break

            # find max tidx
            max_tidx = -1
            for v in visits:
                if max_tidx < v[-1][1]:
                    max_tidx = v[-1][1]

            for v in visits:
                while v[-1][1] < max_tidx:
                    v.pop()

            match = True
            target = visits[0][-1][0]
            if is_transforming:
                if not get_handler(self._graph.nodes(data=True)[target]["layer_dict"]["class_name"]).is_transformer(0):
                    visits[0].pop()
                    continue

            if target not in joints:
                visits[0].pop()
                continue

            for v in visits:
                if v[-1][0] != target:
                   match = False
                   break

            if match:
                return target
                
        return None

    def traverse(self,
                 sources=None,
                 inbound=False,
                 node_callbacks=None,
                 neighbor_callbacks=None,
                 sync=True,
                 stopping_condition=None,
                 previsit=None):
        """A general traversal algorithm for NNs.

        if `sync` is True, it traverses a neural network graph in similar order of NN execution.
        The similar order formally means that the traversing algorithm is designed to push a node n into the traversing stack,
        only if all the inbound neighbors of n are already popped from the stack.

        For example, a node (layer) v is supposed to be concatenation.
        There can be multiple inbounding nodes to v.
        We want to sure that all the inbounding nodes are already processed before processing v like NN execution.
        That why I mentioned the similar order.
       
        # Arguments.
            sources: a list of staring nodes.
            inbound: bool, the flag for indicating search direction.
            node_callbacks: a list, it is a list of functions (name, level) called with the name of a popped node and its traversing level.
            neighbor_callbacks: a list, it is a list of functions called with edge `e`.
            sync: bool, if it is True, the traversal is done in similar oder of NN execution. Otherwise, it is a normal DFS.
            stopping_condition: a function(e), it lets us know whether we need to expand the traversal over edge e.

        """

        visit = []
        if previsit is None:
            visit_ = set()
        else:
            visit_ = previsit

        # if sources is None, then we start from leaves.
        if sources is None:
            sources = [x for x in self._graph.nodes(data=True)
                if (inbound and self._graph.out_degree(x[0])==0)
                    or (not inbound and self._graph.in_degree(x[0])==0)]
        stk = []
        for s in sources:
            if s[1]["nlevel"] == 0 or s[1]["nlevel"] == 1:
                stk.append((s[0], 0))
            else:
                for idx in range(s[1]["nlevel"]):
                    stk.append((s[0], idx))

        if sync and not inbound:
            dependency = {
                n[0] : [
                    [0 for _ in range(len(flow))]
                    for flow in n[1]["layer_dict"]["inbound_nodes"]
                ]
                for n in self._graph.nodes(data=True) if "inbound_nodes" in n[1]["layer_dict"]
            }
        while len(stk) > 0:
            curr, level = stk.pop()
            visit.append((curr, level))

            if node_callbacks is not None:
                for callback in node_callbacks:
                    callback(curr, level)

            if stopping_condition is not None and stopping_condition((curr, level), is_edge=False):
                break

            neighbors = self._graph.in_edges(curr, data=True) if inbound else self._graph.out_edges(curr, data=True)

            for e in neighbors:
                src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
                if neighbor_callbacks is not None:
                    for callback in neighbor_callbacks:
                        callback(e)

                if stopping_condition is not None and stopping_condition(e, is_edge=True):
                    continue

                if inbound:
                    if level_change[1] == level and (src, level_change[0]) not in visit_:
                        stk.append((src, level_change[0]))
                        visit_.add((src, level_change[0]))
                else: #outbound
                    if sync:
                        dependency[dst][level_change[1]][inbound_idx] = 1
                        if np.sum(dependency[dst][level_change[1]]) < len(dependency[dst][level_change[1]]):
                            continue
                    if level_change[0] == level and (sync or (dst, level_change[1]) not in visit_):
                        stk.append((dst, level_change[1]))
                        visit_.add((dst, level_change[1]))
        return visit

    def parse(self):
        """Parse a given network. 

        """
        model_dict = json.loads(self._model.to_json())
        self._model_dict = model_dict
        layers = model_dict["config"]["layers"]

        # Load nodes and edges onto an internal graph defined by networkx.
        for layer in layers: 
            if "inbound_nodes" not in layer: # InputLayer
                self._graph.add_node(layer["config"]["name"], layer_dict=layer, nlevel=0)
            else:
                self._graph.add_node(layer["config"]["name"], layer_dict=layer, nlevel=len(layer["inbound_nodes"]))

        for layer in layers:
            if "inbound_nodes" not in layer: # InputLayer
                continue

            for flow_idx, flow in enumerate(layer["inbound_nodes"]):
                if type(flow[0]) != list:
                    inbound = flow
                    src = inbound[0]
                    dst = layer["config"]["name"]
                    self._graph.add_edge(
                        src, dst, level_change=(inbound[1], flow_idx), tensor=inbound[2], inbound_idx=0, temp=inbound[-1])
                else:
                    for in_idx, inbound in enumerate(flow):
                        src = inbound[0]
                        dst = layer["config"]["name"]
                        self._graph.add_edge(
                            src, dst, level_change=(inbound[1], flow_idx), tensor=inbound[2], inbound_idx=in_idx, temp=None)

        v = self.traverse()
        self.torder = {
            name:idx
            for idx, (name, _) in enumerate(v)
        }

    def get_topology(self):
        """Return a networkx graph having the topology of the graph.

        # Returns.
            A graph (networkx)

        """
        return copy.deepcopy(self._graph)


    def get_leaves(self, block):
        block_set = set([l.name for l in block])
        input_leaves = []
        output_leaves = []
        for l in block:
            is_input_leaf = True
            is_output_leaf = False
            for e in self._graph.in_edges(l.name, data=True): # inbound
                src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
                if src in block_set:
                    is_input_leaf = False
                    break
            if not is_input_leaf:
                is_output_leaf = True
                for e in self._graph.out_edges(l.name, data=True): # inbound
                    src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
                    if dst in block_set:
                        is_output_leaf = False
                        break

            if is_input_leaf:
                input_leaves.append(l.name)
            elif is_output_leaf:
                output_leaves.append(l.name)

        return input_leaves, output_leaves

    def get_subnet(self, block, model, target_shapes=None):
        # block: list of names
        block_set = set([l.name for l in block])
        inputs, outputs = self.get_leaves(block)

        # check `inputs` are sufficient to find `outputs`
        new_inputs = []
        def check(n, level):
            if n not in block_set and n not in new_inputs:
                new_inputs.append(n)
        def stop_check(e, is_edge):
            if not is_edge:
                return False
            if e[1] in new_inputs or e[1] in inputs:
                return True
            return False

        sources_for_check = [ x for x in self._graph.nodes(data=True) if x[1]["layer_dict"]["config"]["name"] in outputs ]
        v = self.traverse(sources=sources_for_check, node_callbacks=[check], stopping_condition=stop_check, inbound=True)
        for v_ in v:
            if v_[0] not in block_set:
                block_set.add(v_[0])
                block.append(model.get_layer(v_[0]))
        inputs += new_inputs

        layers = {
            l.name:l for l in block
        }
        for new in new_inputs:
            layers[new] = model.get_layer(new)

        # define input_tensor
        in_tensors = {}
        for i, in_ in enumerate(inputs):
            """
            if type(layers[in_].input) == list:
                in_tensors[in_] = []
                for i_ in layers[in_].input:
                    if i_.shape[0] is None:
                        in_tensors[in_].append(tf.keras.Input(i_.shape[1:]))
                    else:
                        in_tensors[in_].append(tf.keras.Input(i_.shape))
            else:
                if layers[in_].input.shape[0] is None:
                    in_tensors[in_] = tf.keras.Input(shape=layers[in_].input.shape[1:])
                else:
                    in_tensors[in_] = tf.keras.Input(shape=layers[in_].input.shape)
            """
            shape = list(layers[in_].output.shape)
            if target_shapes is not None:
                shape[1] = target_shapes[i][1]
                shape[2] = target_shapes[i][2]
            if layers[in_].output.shape[0] is None:
                shape = shape[1:]
                in_tensors[in_] = tf.keras.Input(shape=shape, name=in_+"_inputlayer")
            else:
                in_tensors[in_] = tf.keras.Input(shape=shape, name=in_+"_inputlayer")

        out_tensors = {}

        def callback_(n, level):
            if n in in_tensors:
                assert n not in out_tensors
                out_tensors.update({
                    (n,level):in_tensors[n]
                })
            else: 
                inputs_ = []
                for e in self._graph.in_edges(n, data=True):
                    src, dst, level_change, inbound_idx, tidx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"], e[2]["tensor"]
                    if level_change[1] == level:
                        ot = out_tensors[(src, level_change[0])]
                        if type(ot) == tuple:
                            ot = ot[tidx]
                        inputs_.append(ot)
                if len(inputs_) == 1:
                    inputs_ = inputs_[0]

                out_tensors.update({
                    (n,level):layers[n](inputs_)
                })

        def stop_cond(e, is_edge):
            if not is_edge:
                return False
            src, dst, level_change, inbound_idx = e[0], e[1], e[2]["level_change"], e[2]["inbound_idx"]
            if dst not in block_set:
                return True
            if src in outputs:
                return True
            return False

        sources = [ x for x in self._graph.nodes(data=True) if x[1]["layer_dict"]["config"]["name"] in inputs ]
        self.traverse(sources=sources, node_callbacks=[callback_], stopping_condition=stop_cond)

        #TODO
        outputs_ = [
            out_tensors[(out, 0)] for out in outputs
        ]
        inputs_ = [
            in_tensors[in_] for in_ in inputs
        ]
        output_model = keras.Model(inputs=inputs_, outputs=outputs_)

        json_ = output_model.to_json()
        output_model_ = tf.keras.models.model_from_json(json_)
        output_model_.set_weights(output_model.get_weights())
        return output_model_, inputs, outputs

if __name__ == "__main__":

    from tensorflow import keras
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
    #model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)

    tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

    parser = NNParser(model)
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
