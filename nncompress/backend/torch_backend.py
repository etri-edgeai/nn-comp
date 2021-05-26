from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_type(cls_name):
    from mlcorekit.backend import torch_
    if hasattr(torch_, cls_name):
        return getattr(torch_, cls_name)
    else:
        raise NotImplementedError

def cast(x, dtype=np.float32):
    if type(dtype) == str:
        return x.to(dtype=getattr(torch, dtype))
    else:
        return x.to(dtype=dtype)

def function(func, *args, **kwargs):
    if hasattr(torch, func):
        f = getattr(torch, func)
    elif hasattr(F, func):
        f = getattr(F, func)
    else:
        raise NotImplementedError("`%s` is not supported." % func)
    assert(callable(f))
    return f(*args, **kwargs)

def get_out_channel_idx():
    return 0

def floor(x):
    return torch.floor(x)

def round(x):
    return torch.round(x)

def sum(x):
    return torch.sum(x)

def norm(x, p):
    return torch.norm(x, p=p)

def cmul(data, mask):
    # N C W H
    data = data.transpose(1, -1)
    return (data * mask).transpose(-1, 1)

def concat(x, y, dim=0):
    return torch.cat(x, y, dim=dim)
