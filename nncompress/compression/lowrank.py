from __future__ import absolute_import
from __future__ import print_function

import tensorly as tl
from tensorly.decomposition import partial_tucker
from scipy.sparse.linalg import svds
import numpy as np

from nncompress import backend as M

def tucker(tensor, in_rank, out_rank):
    tl_tensor = tl.tensor(tensor)
    d = partial_tucker(tl_tensor, [2, 3], [in_rank, out_rank], init='svd')
    c = d[0] # core
    u = d[1][0][np.newaxis, np.newaxis, ...]
    vt = np.transpose(d[1][1])[np.newaxis, np.newaxis, ...]

    c = c.astype(tensor.dtype)
    u = u.astype(tensor.dtype)
    vt = vt.astype(tensor.dtype)
    return [u, c, vt]

def svd(mat, rank):
    u, s, vt = svds(mat, rank)
    u = u.astype(mat.dtype)
    s = s.astype(mat.dtype)
    vt = vt.astype(mat.dtype)
    return [u, s, vt]

def decompose(model, targets, custom_objects=None):
    """Compress a model written in tf.Keras or PyTorch.

    In case of PyTorch, `model` must be a layer.

    """
    decomposed = []
    targets_ = []
    for target, ratio in targets:
        weights = M.get_weights(model, target)
        rank = min(int(ratio * weights[0].shape[-1]), int(ratio * weights[0].shape[-2]))
        if rank < 3:
            continue
        if len(weights[0].shape) == 4:
            if type(rank) == tuple or type(rank) == list:
                d = tucker(weights[0], rank[0], rank[1])
            else:
                d = tucker(weights[0], rank, rank)
        elif len(weights[0].shape) == 2:
            d = svd(weights[0], rank)
        if len(weights) > 1:
            d.append(weights[1]) # bias
        decomposed.append(d)
        targets_.append(target)
    return M.decompose(model, targets_, decomposed, custom_objects=custom_objects)

if __name__ == "__main__":
    
    import numpy as np 
    
    t = np.random.rand(4,4,32,32)
    c, u, vt = tucker(t, 12, 12)

    m = np.random.rand(12,24)
    u, s, vt = svd(m, 10)
