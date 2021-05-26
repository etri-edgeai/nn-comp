from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncompress.assets.formula.gate import b, gate_func
from nncompress import backend as M

def get_mask(idx_array, val, L=10e8, grad_shape_func=None):
    if callable(grad_shape_func):
        return (idx_array < val).to(dtype=torch.float32) + ((L * val - M.floor(L * val)) / L) * grad_shape_func(val)
    elif grad_shape_func is not None:
        return (idx_array < val).to(dtype=torch.float32) + ((L * val - M.floor(L * val)) / L) * M.function(grad_shape_func, val)
    else:
        return (idx_array < val).to(dtype=torch.float32) + ((L * val - M.floor(L * val)) / L)

class DifferentiableEmbedding(nn.Module):
    """Differentiable Embedding Implementation

    """
    def __init__(self,
                 vocab_size,
                 output_dim,
                 grad_shape_func=torch.tanh,
                 init_func=nn.init.uniform_):
        super(DifferentiableEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.FloatTensor(int(vocab_size), int(output_dim)), requires_grad=True)
        self.gates = nn.Parameter(torch.FloatTensor(int(vocab_size),), requires_grad=True)
        self.grad_shaping = grad_shape_func

        self.index_array = torch.FloatTensor([ i for i in range(output_dim)]).to("cuda:0")

        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # Init
        init_func(self.gates, b=output_dim)
        init_func(self.embedding)

    def report(self):
        for i in range(self.vocab_size):
            if i % 1000 == 0:
                print(self.gates[i])

    def forward(self, input):
        bags = []
        for bidx, data in enumerate(input):
            vecs = []
            for idx in data:
                idx = int(idx)
                vec = self.embedding[idx]
                # Apply differentiable mask
                mask = get_mask(self.index_array, self.gates[idx], grad_shape_func=self.grad_shaping)
                vecs.append(vec * mask)
            bags.append(torch.stack(vecs))
        return torch.stack(bags)
