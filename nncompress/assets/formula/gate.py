from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from nncompress.assets.formula.formula import Formula
from nncompress import backend as M

class SimplePruningGateFormula(Formula):

    def __init__(self):
        super(SimplePruningGateFormula, self).__init__()

    def compute(self, input):
        return M.cmul(input, self.binary_selection())

    def binary_selection(self):
        return M.round(self.gates)  # gates consists of ones or zeros.

    def get_sparsity(self):
        selection = self.binary_selection()
        return 1.0 - M.sum(selection) / self.gates.shape[0]
