from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mlcorekit import backend as M

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

def prune_filter(model, domain, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None):
    return M.prune_filter(model, domain, mode, custom_objects)  
 
def prune(model, targets, mode="channel", method="magnitude", sample_inputs=None, custom_objects=None):
    """Compress a model written in tf.Keras. or PyTorch.

    For PyTorch, it only supports weight pruning.

    """
    masking = []
    replace_mappings = []
    for target, ratio in targets:
        replace_mappings.append((target, [target]))
        w = M.get_weights(model, target)[0]
        if mode == "channel":
            if ratio * w.shape[-1] < 3:
                continue
        if method == "magnitude":
            mask = _magnitude_based_mask(w, ratio, mode)
        elif callable(method):
            mask = method(w, ratio, mode, model)
        else:
            raise NotImplementedError("%s is not implemented." % method)
        masking.append((target, mask))

        if sample_inputs is not None:
            for data in sample_inputs:
                pass
    model_, history = M.prune(model, masking, mode=mode, custom_objects=custom_objects)
    return model_, replace_mappings, history
