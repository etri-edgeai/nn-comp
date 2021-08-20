""" nnmm """

class NNMM(object):

    def __init__(self, model=None, handler=None):
        self._model = model
        self._handler = handler
        self._index = None

    def build(self, search_space):
        """ build """
        self._index.build(self._model, search_space)

    def train(self):
        """ sub-network training"""
        pass

    def evaluate(self):
        """ sub-network evaluation"""
        pass

    def retrieve(self, spec):
        """ sub-network retrieval"""
        pass
