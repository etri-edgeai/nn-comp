""" Formula Base """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

from nncompress import backend as M

class Formula(ABC):
    """ Formula """

    @abstractmethod
    def compute(self, *input, training=False):
        """Compute the result for the given input.

        # Arguments
            input: input(Backend-aware Tensor).
            training: bool, a flag for identifying training.

        # Returns
            Result Tensor(s).

        """
