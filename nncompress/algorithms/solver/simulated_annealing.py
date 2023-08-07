""" Simulated Annealing Implementation

"""
from __future__ import absolute_import
from __future__ import print_function

import random
import math
import time
import numpy as np

from nncompress.algorithms.solver.solver import Solver

INIT_TEMP = 1000000
DELAY = 1
def temperature(iter_, max_niters, curr_temp=-1, init_temp=INIT_TEMP, delta_temp=1000, use_delay=True):
    """ Simple temperature function

    """
    if curr_temp != -1 and use_delay:
        iter_ = iter_ // DELAY
    return init_temp - iter_ * delta_temp * (init_temp / (max_niters * delta_temp))

def transition_prob(diff_score, temp, init_temp=INIT_TEMP):
    """ Transition prob.

    """
    if diff_score > 0:
        return 1.0
    else:
        return math.exp(50000000 * diff_score / temp)

class SimulatedAnnealingSolver(Solver):
    """ Simulated Annealing Solver """

    def __init__(self, score_func, max_niters, temp_func=temperature, tprob_func=transition_prob):
        super(SimulatedAnnealingSolver, self).__init__(score_func)
        self.max_niters = max_niters
        self._temp_func = temperature
        self._best = None
        self._best_score = None
        print(time.ctime(time.time()))

    def solve(self, initial_state, callbacks=None):
        """ Solve function """
        state = initial_state
        self._best = state
        T = -1
        for i in range(self.max_niters):
            T = self._temp_func(i, self.max_niters, T)
            score = self._score_func(state)
            new_state = state.get_next()
            new_score = self._score_func(new_state)
            if hasattr(new_state, "report"):
                new_state.report()

            while new_score == 0.0:
                new_state = state.get_next()
                new_score = self._score_func(new_state)

            if self._best_score is None:
                self._best_score = self._score_func(self._best)

            if new_score > self._best_score:
                self._best = new_state
                self._best_score = new_score

            prob = transition_prob(new_score - score, T)
            print("[%s] %d iterations, Score:%.4f   New score:%.4f Best score:%.4f  prob:%.4f, output:%s" %\
                (time.ctime(time.time()), i, score, new_score, self._best_score, prob, str(self._best)))
            transition = False
            if prob >= random.random():
                state = new_state
                transition = True

            if callbacks is not None:
                for c in callbacks:
                    c(state, i, transition)
        return state, self._best
