import networkx as nx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np

from .cct import StochasticDepth

class WeightedSum(tf.keras.layers.Layer):

    def __init__(
        self,
        n,
        drop_path_rate=-1,
        name=None):
        super(WeightedSum, self).__init__(name=name)

        self.n = n
        self.drop_path_rate = drop_path_rate

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(self.n,), dtype="float32", trainable=True, initializer="ones")

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
            "drop_path_rate":self.drop_path_rate,
            "name":self.name
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


def base_node_func(x, dim, graph, node, init=tf.keras.initializers.GlorotUniform(), l2_reg=1e-4, dropout_rate=0.1):
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
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg))(x)

    else:
        x = tf.keras.layers.SeparableConv2D(
            dim,
            (3,3),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    se_ratio = 0.25
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(dim * se_ratio))
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        #if bn_axis == 1:
        #    se = tf.keras.layers.Reshape((dim, 1, 1))(se)
        #else:
        se = tf.keras.layers.Reshape((1, 1, dim))(se)
        se = tf.keras.layers.Conv2D(filters_se, 1,
                           padding='same',
                           kernel_initializer=init)(se)
        se = tf.keras.layers.ReLU()(se)
        se = tf.keras.layers.Conv2D(dim, 1,
                           padding='same',
                           kernel_initializer=init)(se)
        se = tf.keras.layers.Activation("sigmoid")(se)

        x = tf.keras.layers.multiply([x, se])

    return x

def mlp(x, hidden_units, dropout_rate, activations):
    for units, activation in zip(hidden_units, activations):
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
            bias_initializer='zeros')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def transformer_node_func(x, dim, graph, node, dpr, num_heads=4, init=tf.keras.initializers.GlorotUniform(), l2_reg=1e-4, dropout_rate=0.1):
    if len(x) > 1:
        x = WeightedSum(len(x))(x)
    else:
        x = x[0]

    if len(x.shape) > 3:
        x = tf.keras.layers.ReLU()(x)

        #x = tf.keras.layers.ZeroPadding2D(2)(x)
        x = tf.keras.layers.MaxPool2D(3, 2, "same")(x)

        encoded_patches = tf.reshape(
                x,
                (-1, tf.shape(x)[1] * tf.shape(x)[2], tf.shape(x)[-1]),
        )
        seq_length = int(x.shape[1]) * int(x.shape[2])

        pos_embed = tf.keras.layers.Embedding(
            input_dim=seq_length, output_dim=dim
        )

        #positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = np.arange(seq_length)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings
    else:
        encoded_patches = x

    mlp_ratio = 2.0
    transformer_activations = [
        tf.nn.gelu,
        None
    ]
    transformer_units = [
        dim*mlp_ratio,
        dim
    ]

    # Calculate Stochastic Depth probabilities.

    # Layer normalization 1.
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

    # Create a multi-head attention layer.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim, dropout=dropout_rate
    )(x1, x1)

    # Skip connection 1.
    attention_output = StochasticDepth(dpr)(attention_output)
    x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2)

    # MLP.
    x3 = mlp(x2, hidden_units=transformer_units, dropout_rate=0.0, activations=transformer_activations)

    # Skip connection 2.
    x3 = StochasticDepth(dpr)(x3)
    encoded_patches = tf.keras.layers.Add()([x3, x2])
    return encoded_patches


def make_nn_from_graph(
    g,
    dim,
    input_shape=None,
    input_tensor=None,
    #node_func=base_node_func,
    node_func=transformer_node_func,
    identifier=None,
    ret_model=True,
    l2_reg=1e-4,
    dropout_rate=0.1):
    """g is a valid graph.

    """

    if node_func == transformer_node_func:
        stochastic_depth_rate = 0.1
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, g.number_of_nodes())]

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
    idx = 0
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
            if node_func == transformer_node_func:
                out = node_func(inputs[curr], dim, g, curr, dpr[idx], l2_reg=l2_reg, dropout_rate=dropout_rate)
                idx += 1
            else:
                out = node_func(inputs[curr], dim, g, curr, l2_reg=l2_reg, dropout_rate=dropout_rate)
    
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


def make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess, node_func):
    o_net_in = tf.keras.Input(input_shape)
    net_in = preprocess(o_net_in)

    # make input/ouptut nodes
    for idx, g in enumerate(graphs):

        dim = regime[idx]
        if idx == 0:
            net_in_ = net_in
        else:
            net_in_ = net_out

        net_out = make_nn_from_graph(g[0], dim, input_tensor=net_in_, identifier=g[1], ret_model=False, node_func=node_func)

    net_out = postprocess(net_out) # classifier
    return tf.keras.Model(inputs=o_net_in, outputs=net_out)


def randwired_cct(input_shape=(32,32,3), dim=64, nstages=3, num_nodes=7, l2_reg=0.0, dropout_rate=0.1, num_classes=100):

    regime = [dim * pow(2, i) for i in range(nstages)]
    #init = tf.keras.initializers.HeNormal()
    init = tf.keras.initializers.GlorotUniform()

    """
    tf.keras.layers.SeparableConv2D(
        dim,
        (3,3),
        padding="same",
        depthwise_initializer=init,
        pointwise_initializer=init,
        bias_initializer="zeros",
        pointwise_regularizer=regularizers.l2(l2_reg),
        depthwise_regularizer=regularizers.l2(l2_reg)),
    """

    """
    preprocess = tf.keras.Sequential([
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            dim,
            (3,3),
            padding="same",
            use_bias=False,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        tf.keras.layers.ReLU()
    ])
    """

    def preprocess(x):
        x = tf.keras.layers.Conv2D(
            dim,
            (3,3),
            padding="same",
            use_bias=False,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        return x

    # postprocess
    def postprocess(encoded_patches):
        representation = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
        attention_weights = tf.nn.softmax(
            tf.keras.layers.Dense(1,
                        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
                        bias_initializer="zeros"
            )(representation),
            axis=1
        )
        weighted_representation = tf.matmul(
            attention_weights, representation, transpose_a=True
        )
        weighted_representation = tf.squeeze(weighted_representation, -2)

        # Classify outputs.
        logits = tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None),
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(l2_reg))(weighted_representation)

        return logits

    p, k, m = 0.75, 4, 5
    graphs = []
    identifier = "G0"
    #g = nx.random_graphs.barabasi_albert_graph(num_nodes, m)
    g = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, k, p)
    g = make_valid_graph(g, identifier)
    graphs.append((g, identifier))

    model = make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess, node_func=transformer_node_func)

    print("### num of params:", model.count_params())
    tf.keras.utils.plot_model(model, to_file="model.png", expand_nested=True)
    return model

def randwired_cifar(input_shape=(32,32,3), dim=78, nstages=3, num_nodes=32, l2_reg=1e-4, dropout_rate=0.1, num_classes=100):

    regime = [dim * pow(2, i) for i in range(nstages)]
    #init = tf.keras.initializers.HeNormal()
    init = tf.keras.initializers.GlorotUniform()

    def preprocess(x):
        x = tf.keras.layers.SeparableConv2D(
            dim // 2,
            (3,3),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.ReLU()(x)

    def postprocess(x):
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.SeparableConv2D(
            1280,
            (1,1),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        return tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg))(x)


    """
    preprocess = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(
            dim // 2,
            (3,3),
            padding="same",
            depthwise_initializer=init,
            pointwise_initializer=init,
            bias_initializer="zeros",
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
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
            pointwise_regularizer=regularizers.l2(l2_reg),
            depthwise_regularizer=regularizers.l2(l2_reg)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg))
    ])
    """

    p, k, m = 0.75, 4, 5
    graphs = []
    for idx in range(len(regime)):
        identifier = "G%d" % idx
        #g = nx.random_graphs.barabasi_albert_graph(num_nodes, m)
        g = nx.random_graphs.connected_watts_strogatz_graph(num_nodes, k, p)
        g = make_valid_graph(g, identifier)
        graphs.append((g, identifier))

    model = make_nn_from_graphs(graphs, regime, input_shape, preprocess, postprocess, node_func=base_node_func)

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
