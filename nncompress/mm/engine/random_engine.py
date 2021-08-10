from __future__ import absolute_import
from __future__ import print_function

from nncompress.mm.engine.engine import Engine
from nncompress.backend.tensorflow_.transformation.parser import NNParser

class RandomExplorationEngine(Engine):

    def __init__(self, model, custom_objects=None):
        self._parser = NNParser(model, custom_objects=custom_objects)
        self._parser.parse()
 
    def build(self, search_space):
        """Build the index for this engine.

        """
        # Read Search Space


        # Random Search
