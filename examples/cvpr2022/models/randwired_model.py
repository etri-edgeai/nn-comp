import networkx as nx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

dropout_rate = 0.2
l2_reg= 1e-4

class WeightedSum(tf.keras.layers.Layer):

    def __init__(
        self,
        n,
        drop_path_rate=-1):
        super(WeightedSum, self).__init__()

        self.n = n
        self.drop_path_rate = drop_path_rate

    def build(self, input_shape):
        #self.w = self.add_weight("w", shape=(self.n,), dtype="float32", trainable=True, initializer=tf.keras.initializers.RandomUniform(minval=-2, maxval=2.))
        self.w = self.add_weight("w", shape=(self.n,), dtype="float32", trainable=True, initializer="ones")
        #self.w = self.add_weight("w", shape=(self.n,), dtype="float32", trainable=False, initializer=tf.keras.initializers.Ones())

    def call(self, inputs, training=None):
        output_ = tf.stack(inputs, axis=-1)
        w_ = tf.math.sigmoid(self.w)
        if self.drop_path_rate != -1 and training:
            rand_select = tf.random.uniform(shape=w_.shape)
            rand_select = tf.cast(rand_select > self.drop_path_rate, dtype=tf.float32)
            w_ = w_ * rand_select
        else:
            w_ = w_

        return tf.math.reduce_sum(output_ * w_, axis=-1)

    def get_config(self):
        return {
            "n":self.n,
            "drop_path_rate":self.drop_path_rate
        }

def get_leaves(g, inputs):
    
    stk = [ node for node in inputs ]
    visit = set(stk)
    outputs = set()
    while len(stk) != 0:
        curr = stk.pop()

        is_leaf = True
        for edge in g.edges(curr):
            _, tar = edge

            if tar not in visit:
                visit.add(tar)
                stk.append(tar)
                is_leaf = False

        if is_leaf:
            outputs.add(curr)

    return list(outputs)


def make_valid_graph(g, identifier, is_undirected=True):

    # undirected to directed
    if is_undirected:
        g_ = nx.DiGraph()
        for n in g:
            g_.add_node(n)

        for n in g:
            for neighbor in g.adj[n]:
                if n < neighbor:
                    g_.add_edge(n, neighbor)
        g = g_

    inputs = []
    for n in g.nodes:
        if g.in_degree(n) == 0:
            inputs.append(n)
    outputs = []
    for n in g.nodes:
        if g.out_degree(n) == 0:
            outputs.append(n)
    assert len(outputs) > 0

    g_ = g.copy()
    attrs = {
        node: {"type":"NORMAL"}
        for node in g_
    }
    nx.set_node_attributes(g_, attrs)

    input_node = identifier+"_INPUT"
    output_node = identifier+"_OUTPUT"
    g_.add_nodes_from([
        (input_node, {"type":"INPUT"}),
        (output_node, {"type":"OUTPUT"})
    ])

    for input_ in inputs:
        g_.add_edge(input_node, input_)
    for output_ in outputs:
        g_.add_edge(output_, output_node)
    
    return g_


def base_node_func(x, dim, graph, node, init=tf.keras.initializers.GlorotUniform()):
    """node_type is not used.

    """
    if len(x) > 1:
        x = WeightedSum(len(x))(x)
    else:
        x = x[0]
    x = tf.keras.layers.ReLU()(x)

    in_edges = [ e for e in graph.in_edges(node) ]

    if graph.in_degree(node) == 1 and graph.nodes[in_edges[0][0]]["type"] == "INPUT":
        x = tf.keras.layers.SeparableConv2D(
            dim,
            (3,3),
            strides=(2,2),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg))(x)
    else:
        x = tf.keras.layers.SeparableConv2D(
            dim,
            (3,3),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def make_nn_from_graph(g, dim, input_shape=None, input_tensor=None, node_func=base_node_func, identifier=None, ret_model=True):
    """g is a valid graph.

    """

    # find input node
    input_node = None
    for node in g:
        if g.nodes[node]["type"] == "INPUT":
            input_node = node
            break
    assert input_node is not None

    stk = [input_node]
    inputs = {
        n:[] for n in g
    }

    if input_tensor is not None:
        net_in = input_tensor
    else:
        net_in = tf.keras.Input(shape=input_shape)
    net_out = None
    visit = set()
    while len(stk) != 0:
        curr = stk.pop()

        if g.in_degree(curr) != len(inputs[curr]): # delayed DFS
            continue

        if curr in visit:
            continue
        visit.add(curr)

        # handling nodes upon their types
        if g.nodes[curr]["type"] == "INPUT":
            out = net_in
        elif g.nodes[curr]["type"] == "OUTPUT":
            if len(inputs[curr]) == 1:
                net_out = inputs[curr][0]
            else:
                net_out = tf.keras.layers.Average()(inputs[curr])
        else:
            out = node_func(inputs[curr], dim, g, curr)
    
        # transfer the output to the next nodes as input.
        for edge in g.edges(curr):
            _, tar = edge
            inputs[tar].append(out)
            stk.append(tar)

    if ret_model:
        if identifier is not None:
            return tf.keras.Model(inputs=net_in, outputs=net_out, name=identifier)
        else:
            return tf.keras.Model(inputs=net_in, outputs=net_out)
    else:
        return net_out


def make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess):

    o_net_in = tf.keras.Input(input_shape)
    net_in = preprocess(o_net_in)

    # make input/ouptut nodes
    for idx, g in enumerate(graphs):

        dim = regime[idx]
        if idx == 0:
            net_in_ = net_in
        else:
            net_in_ = net_out

        net_out = make_nn_from_graph(g[0], dim, input_tensor=net_in_, identifier=g[1], ret_model=False)
    net_out = postprocess(net_out) # classifier
 
    return tf.keras.Model(inputs=o_net_in, outputs=net_out)

def randwired_cifar(num_classes):

    dim = 78
    input_shape = (32,32,3)
    regime = [dim, dim*2, dim*4]
    num_nodes = 32
    #init = tf.keras.initializers.HeNormal()
    init = tf.keras.initializers.GlorotUniform()

    preprocess = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            dim // 2,
            (3,3),
            padding="same",
            kernel_initializer=init,
            kernel_regularizer=regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),
        tf.keras.layers.SeparableConv2D(
            dim,
            (3,3),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

    postprocess = tf.keras.Sequential([
        #tf.keras.layers.Conv2D(1280, (1,1), padding="same", kernel_initializer=init),
        tf.keras.layers.ReLU(),
        tf.keras.layers.SeparableConv2D(
            1280,
            (1,1),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg))
    ])

    p, k, m = 0.75, 4, 5
    graphs = []
    for idx in range(len(regime)):
        identifier = "G%d" % idx
        #g = nx.random_graphs.barabasi_albert_graph(num_nodes, m)
        g = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, k, p)
        g = make_valid_graph(g, identifier)
        graphs.append((g, identifier))

    model = make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess)

    print("### num of params:", model.count_params())

    tf.keras.utils.plot_model(model, to_file="model.png", expand_nested=True)
    return model


def test():

    dim = 78
    input_shape = (32,32,3)
    regime = [dim*2, dim*4, dim*8]
    num_nodes = 32

    preprocess = tf.keras.Sequential([
        tf.keras.layers.Conv2D(dim, (3,3), padding="same"),
        tf.keras.layers.BatchNormalization()
    ])

    postprocess = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1280, (1,1), padding="same"),
        tf.keras.layers.BatchNormalization()
    ])

    p, k, m = 0.75, 4, 5
    graphs = []
    for idx in range(len(regime)):
        identifier = "G%d" % idx
        g = nx.random_graphs.barabasi_albert_graph(num_nodes, m)
        g = make_valid_graph(g, identifier)
        graphs.append((g, identifier))

    model = make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess)

    tf.keras.utils.plot_model(model, to_file="model.png", expand_nested=True)

if __name__ == "__main__":
    test()
