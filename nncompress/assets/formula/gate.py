from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nncompress.assets.formula.formula import Formula
from nncompress import backend as M

def b(x):
    # Assume that > operator must be supported in backends.
    return M.cast((x >= 0.5), "float32")

def gate_func(x, L=10e5, grad_shape_func=None):
    x_ = x - 0.5
    if callable(grad_shape_func):
        return b(x) + ((L * x_ - M.floor(L * x_)) / L) * grad_shape_func(x_)
    elif grad_shape_func is not None:
        return b(x) + ((L * x_ - M.floor(L * x_)) / L) * M.function(grad_shape_func, x_)
    else:
        return b(x) + ((L * x_ - M.floor(L * x_)) / L)

class DifferentiableGateFormula(Formula):

    def __init__(self):
        super(DifferentiableGateFormula, self).__init__()

    def compute(self, input, prefix_gate=None, training=False):
        if prefix_gate is not None:
            if training:
                return M.cmul(input, M.concat(prefix_gate, self.diff_selection()))
            else:
                return M.cmul(input, M.concat(prefix_gate, self.binary_selection()))
        else:
            if training:
                return M.cmul(input, self.diff_selection())
            else:
                return M.cmul(input, self.binary_selection())

    def binary_selection(self):
        return b(self.gates)
    
    def diff_selection(self):
        return gate_func(self.gates, grad_shape_func=self.grad_shaping)

    def selection(self, training):
        if not training:
            return self.binary_selection()
        else:
            return self.diff_selection()

    def get_sparsity(self, training=False):
        selection = self.selection(training)
        return 1.0 - M.sum(selection) / self.gates.shape[0]

    def get_sparsity_loss(self):
        return self.reg_weight * M.norm(self.sparsity - self.get_sparsity(True), 2)
