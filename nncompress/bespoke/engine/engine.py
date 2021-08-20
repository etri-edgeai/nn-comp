from __future__ import absolute_import
from __future__ import print_function

from abc import ABC, abstractmethod

class Engine(ABC):

    @abstractmethod
    def build(self, model, search_space):
        """Build the index for this engine.

        """
