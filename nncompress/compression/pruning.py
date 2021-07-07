from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from nncompress import backend as M
from nncompress.run.projection import extract_sample_features
from nncompress.run.projection import least_square_projection

def cali(model, compressed, masking, helper, nsamples=100, feat_data=None):
    merged_masking = {}
    for layer_name in masking:
        assert masking[layer_name] is not None
        if layer_name not in merged_masking: # leaf
            merged_masking[layer_name] = masking[layer_name]
        else:
            input_mask = None
            output_mask = None
            if masking[layer_name][0] is not None:
                indices = np.where(masking[layer_name][0])
                input_mask = np.copy(masking[layer_name][0])
                input_mask[indices] = merged_masking[layer_name][0]
            if masking[layer_name][1] is  not None:
                indices = np.where(masking[layer_name][1])
                output_mask = np.copy(masking[layer_name][1])
                output_mask[indices] = merged_masking[layer_name][1]
            merged_masking[layer_name] = (input_mask, output_mask)

    layers = []
    for layer_name in merged_masking:
        try:
            layers.append(model.get_layer(layer_name))
        except ValueError: # ignore the case `layer_name` is included in the model.
            continue

    # Layer filtering
    layers_ = []
    for layer in layers:
        if layer.__class__.__name__ == "Conv2D" and layer.kernel_size == (1,1) and layer.strides==(1,1):
            layers_.append(layer)
        elif layer.__class__.__name__ == "Dense":
            layers_.append(layer)
    if len(layers_) == 0:
        return None

    ## apply projection
    if feat_data is None:
        temp_data = extract_sample_features(model, layers_, helper, nsamples=nsamples)
        feat_data = {}
        for layer_name, feat_data_ in temp_data.items():
            feat_data[layer_name] = feat_data_
    least_square_projection(compressed, feat_data, merged_masking)

def _magnitude_based_mask(w, ratio, mode):
    w = np.abs(w)
    if mode == "channel": # output channel
        sum_ = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
        sorted_ = np.sort(sum_, axis=None)
        val = sorted_[int((len(sorted_)-1)*ratio)]
        removal = (sum_ >= val).astype(np.float32)
        return removal
    elif mode == "weight":
        sorted_ = np.sort(w, axis=None)
        val = sorted_[int(len(sorted_)*ratio)]
        w_ = (w >= val).astype(np.float32)
        return w_
    else:
        raise NotImplementedError("`mode` can be 'channel' or 'weight', but %s is given." % mode)

def group_pruning_mask(model, targets, ratio):
    """A magnitude-based pruning method for sharing layers

    """
    sum_ = None
    for t in targets:
        layer = model.get_layer(t)
        w = layer.get_weights()[0]
        if sum_ is None:
            sum_ = np.sum(np.abs(w)/np.max(w), axis=tuple([i for i in range(len(w.shape)-1)]))
        else:
            sum_ += np.sum(np.abs(w)/np.max(w), axis=tuple([i for i in range(len(w.shape)-1)]))
    sorted_ = np.sort(sum_, axis=None)
    val = sorted_[int((len(sorted_)-1)*ratio)]
    removal = (sum_ >= val).astype(np.float32)
    return removal

def random_mask(model, targets, ratio):
    """A magnitude-based pruning method for sharing layers

    """
    t = targets[0]
    w = model.get_layer(t).get_weights()[0]
    sum_ = np.random.rand(w.shape[-1],)
    sorted_ = np.sort(sum_, axis=None)
    val = sorted_[int((len(sorted_)-1)*ratio)]
    removal = (sum_ >= val).astype(np.float32)
    return removal

def weighted_group_pruning_mask(model, targets, ratio):
    """A magnitude-based pruning method for sharing layers

    """
    weight = []
    Ns = []
    sum_ = None
    for idx, t in enumerate(targets):
        layer = model.get_layer(t)
        w = np.abs(layer.get_weights()[0])
        N = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))
        if sum_ is None:
            sum_ = N
        else:
            sum_ += N
        Ns.append(N)
    for idx, N in enumerate(Ns):
        weight.append(N/sum_)

    sum_ = None
    for idx, t in enumerate(targets):
        layer = model.get_layer(t)
        w = np.abs(layer.get_weights()[0])
        N = np.sum(w, axis=tuple([i for i in range(len(w.shape)-1)]))

        if sum_ is None:
            sum_ = np.sum(w/np.max(w), axis=tuple([i for i in range(len(w.shape)-1)])) * weight[idx]
        else:
            sum_ += np.sum(w/np.max(w), axis=tuple([i for i in range(len(w.shape)-1)])) * weight[idx]
    sorted_ = np.sort(sum_, axis=None)
    val = sorted_[int((len(sorted_)-1)*ratio)]
    removal = (sum_ >= val).astype(np.float32)
    return removal

def prune_filter(model, domain, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None):
    return M.prune_filter(model, domain, mode, custom_objects)  
 
def prune(model, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None, helper=None, calibration=False, nsamples=100, feat_data=None):
    """Compress a model written in tf.Keras. or PyTorch.

    For PyTorch, it only supports weight pruning.

    """
    pruned_layers = set()
    masking = []
    replace_mappings = []
    for target, ratio in targets:
        if target in pruned_layers:
            continue

        if mode == "channel":
            sharing_layers = M.get_sharing_layers(model, target)
            for s in sharing_layers:
                assert s not in pruned_layers
                pruned_layers.add(s)
                replace_mappings.append((s, [s]))
            if M.get_weights(model, sharing_layers[0])[0].shape[-1] < 3:
                continue
            if "magnitude" in method:
                if "magnitude_first" == method:
                    w = M.get_weights(model, sharing_layers[0])[0]
                elif "magnitude_last" in method:
                    w = M.get_weights(model, sharing_layers[-1])[0]
                elif "magnitude" == method:
                    w = M.get_weights(model, target)[0]
                else:
                    raise NotImplementedError("%s is not implemented." % method)
                mask = _magnitude_based_mask(w, ratio, mode)
            elif method == "group_sum":
                mask = group_pruning_mask(model, sharing_layers, ratio)
            elif method == "w_group_sum":
                mask = weighted_group_pruning_mask(model, sharing_layers, ratio)
            elif method == "random":
                mask = random_mask(model, sharing_layers, ratio)
            elif callable(method):
                mask = method(model, sharing_layers, ratio)
            else:
                raise NotImplementedError("%s is not implemented." % method)
        else:
            w = M.get_weights(model, target)[0]
            if method == "magnitude":
                mask = _magnitude_based_mask(w, ratio, mode)
            elif callable(method):
                mask = method(w, ratio, mode, model)
            else:
                raise NotImplementedError("%s is not implemented." % method)

        # Even if we only include `target` for channel pruning, the parser will find sharing layers in its internal process.
        masking.append((target, mask))
        if sample_inputs is not None:
            for data in sample_inputs:
                pass
    model_, history = M.prune(model, masking, mode=mode, custom_objects=custom_objects)

    if calibration:
        assert helper is not None
        cali(model, model_, history, helper, nsamples=nsamples, feat_data=feat_data)
    return model_, replace_mappings, history
