from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from nncompress import backend as M

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

def prune_filter(model, domain, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None):
    return M.prune_filter(model, domain, mode, custom_objects)  
 
def prune(model, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None):
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
    return model_, replace_mappings, history
