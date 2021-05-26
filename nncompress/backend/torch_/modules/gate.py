from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from nncompress.assets.formula.gate import DifferentiableGateFormula

class DifferentiableGate(nn.Module, DifferentiableGateFormula):
    """DifferentiableGate Implementation

    """
    def __init__(self,
                 ngates,
                 sparsity,
                 reg_weight=0.5,
                 grad_shape_func=None,
                 init_func=nn.init.uniform_):
        super(DifferentiableGate, self).__init__()
        self.ngates = ngates
        self.gates = nn.Parameter(torch.FloatTensor(ngates,), requires_grad=True)
        self.grad_shaping = grad_shape_func
        self.sparsity = sparsity
        self.reg_weight = reg_weight

        # Init
        init_func(self.gates)

    def forward(self, input):
        return self.compute(input, training=self.training)

class DifferentiableGateWithPrefix(DifferentiableGate):
    """DifferentiableGate Implementation

    """
    def __init__(self,
                 ngates,
                 sparsity,
                 reg_weight=0.5,
                 grad_shape_func=None,
                 init_func=nn.init.uniform_):
        super(DifferentiableGateWithPrefix, self).__init__()
        self.ngates = ngates
        self.gates = nn.Parameter(torch.FloatTensor(ngates,), requires_grad=True)
        self.grad_shaping = grad_shape_func
        self.sparsity = sparsity
        self.reg_weight = reg_weight

        # Init
        init_func(self.gates)

    def forward(self, input, prefix_gate):
        return self.compute(input, prefix_gate, training=self.training)
