from __future__ import absolute_import
from __future__ import print_function

from abc import ABC, abstractmethod

class State(ABC):

    def get_next(self):
        """Find neighbors from the current state

        Returns.
            A list of neighbor states.

        """
        return self.get_next_impl()

    @abstractmethod
    def get_next_impl(self):
        """Implemnetation of get_next

        """

    def __str__(self):
        return "BaseState"

class Solver(ABC):

    def __init__(self, score_func):
        self._score_func = score_func

    @abstractmethod
    def solve(self, initial_state, callbacks=None):
        """Solve the problem.

        """
