
import copy

import numpy as np

from group_fisher import add_gates, compute_positions
from nncompress.algorithms.solver.simulated_annealing import SimulatedAnnealingSolver
from nncompress.algorithms.solver.solver import State

class PositionState(State):

    def __init__(self, positions, torder):
        self._pos = positions
        self._torder = torder
        self._score = None
        self._inv_torder = {}
        self._layers = [
            key for key in self._torder
        ]

        for key, item in self._torder.items():
            self._inv_torder[item] = key

    def get_next_impl(self):

        if (np.random.rand() < 0.3 and len(self._layers) > len(self._pos)) or len(self._pos) < 2:

            rand_idx = int(np.random.rand() * len(self._layers))
            while self._layers[rand_idx] in self._pos or rand_idx == 0:
                rand_idx = int(np.random.rand() * len(self._layers))

            target = self._layers[rand_idx]
            target_idx = -1
            for idx, pos in enumerate(self._pos):
                if self._torder[pos] > self._torder[target]:
                    target_idx = idx
                    break

            new_pos = copy.deepcopy(self._pos)
            if target_idx == -1:
                new_pos.append(target)
            else:
                new_pos.insert(target_idx, target)

        elif np.random.rand() < 0.6 and len(self._pos) > 1:
            
            rand_idx = int(np.random.rand() * len(self._pos))
            new_pos = copy.deepcopy(self._pos)
            new_pos.remove(new_pos[rand_idx])

        else:

            while True:

                rand_idx = int(np.random.rand() * len(self._pos))
                target = self._pos[rand_idx]

                if rand_idx > 0:
                    prev = self._pos[rand_idx-1]
                    prev_trank = self._torder[prev]
                else:
                    prev = None
                    prev_trank = -1

                if rand_idx < len(self._pos) - 1:
                    next_ = self._pos[rand_idx+1]
                    next_trank = self._torder[next_]
                else:
                    next_ = None
                    next_trank = len(self._pos)

                range_ = list(range(prev_trank+1, next_trank))
                if len(range_) > 0:
                    break

            rand_pidx = int(np.random.rand() * len(range_))
            new_trank = range_[rand_pidx]
            new_layer = self._inv_torder[new_trank]
            
            new_pos = copy.deepcopy(self._pos)
            new_pos[rand_idx] = new_layer

        print(rand_idx, [
            (pos, self._torder[pos]) for pos in new_pos
        ])
        return PositionState(new_pos, self._torder)

    def __str__(self):
        return "PositionState"


def find_positions(
    prune_func,
    dataset,
    model,
    model_handler,
    position_mode=1, # initial positions
    with_label=False,
    label_only=False,
    distillation=True,
    num_remove=1,
    enable_distortion_detect=False,
    target_ratio=0.5,
    min_steps=-1,
    period=25,
    n_classes=100,
    num_blocks=-1):

    if num_blocks == -1:
        inc_blocks = True
        num_blocks = 5
    else:
        inc_blocks = False

    gmodel, copied_model, l2g, ordered_groups, torder, parser, _ = add_gates(model, custom_objects=model_handler.get_custom_objects())
    positions = compute_positions(
        copied_model, ordered_groups, torder, parser, position_mode, num_blocks, model_handler.get_heuristic_positions())

    init_state = PositionState(positions, torder)

    def score(state):
        if state._score is not None:
            return state._score
            
        score = prune_func(
            dataset,
            model,
            model_handler,
            state._pos,
            with_label=with_label,
            label_only=label_only,
            distillation=distillation,
            num_remove=num_remove,
            enable_distortion_detect=enable_distortion_detect,
            target_ratio=target_ratio,
            min_steps=min_steps,
            period=period,
            n_classes=n_classes,
            num_blocks=len(state._pos),
            ret_score=True)
        state._score = score
        return score

    solver = SimulatedAnnealingSolver(score, 100)
    state, best = solver.solve(init_state)

    return state._pos, best._pos
