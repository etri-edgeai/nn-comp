from __future__ import print_function

import tensorflow as tf
from sklearn.linear_model import LinearRegression
import numpy as np

def extract_sample_features(model, layers, helper, nsamples=3, npoints=10):
    tensors = []
    for idx, layer in enumerate(layers):
        tensors.append(layer.inbound_nodes[0].input_tensors)
        tensors.append(layer.output)
    model_ = tf.keras.Model(inputs=model.inputs,
                        outputs=tensors)
    sampled_data = helper.sample_training_data(nsamples)
    ret = {}
    for data in sampled_data:
        X, Y = data
        Y_ = model_.predict(X)
        for idx, layer in enumerate(layers):
            layer_input = Y_[2*idx]
            layer_output = Y_[2*idx+1]
            if len(layer_input.shape) == 4:
                #sampling coordinates
                if layer_input.shape[2] == 1:
                    random_X = [0]
                else:
                    random_X = np.random.randint(0, layer_input.shape[2]-1, npoints)

                if layer_input.shape[1] == 1:
                    random_Y = [0]
                else:
                    random_Y = np.random.randint(0, layer_input.shape[1]-1, npoints)

                assert len(random_X) == len(random_Y)
                npoints_ = len(random_X)
                sampled_input = layer_input[:, random_Y, random_X, :].reshape(helper.batch_size * npoints_, -1)
                sampled_output = layer_output[:, random_Y, random_X, :].reshape(helper.batch_size * npoints_, -1)
            else:
                sampled_input = layer_input
                sampled_output = layer_output
            if layer.name not in ret:
                ret[layer.name] = [sampled_input, sampled_output]
            else:
                ret[layer.name][0] = np.vstack((ret[layer.name][0], sampled_input))
                ret[layer.name][1] = np.vstack((ret[layer.name][1], sampled_output))
    return ret

def least_square_projection(model, feature_data, masking):
    for layer_name in feature_data:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue
        if layer_name not in masking:
            continue

        X, Y = feature_data[layer_name]
        if masking[layer_name][0] is not None:
            X = X[:,masking[layer_name][0]]
            reg = LinearRegression(fit_intercept=layer.use_bias)
        reg.fit(X, Y)
        W = reg.coef_.transpose(1,0)
        b = reg.intercept_
        if masking[layer_name][1] is not None:
            W = W[:,masking[layer_name][1]]
            if type(b) == float:
                b = np.ones((Y.shape[-1],)) * b
            b = b[np.array(masking[layer_name][1])]
        if layer.__class__.__name__ == "Conv2D":
            W = W.reshape(1, 1, W.shape[-2], W.shape[-1])
        if layer.use_bias:
            layer.set_weights([W,b])
        else:
            layer.set_weights([W])
