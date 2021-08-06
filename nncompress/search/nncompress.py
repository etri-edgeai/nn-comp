from __future__ import absolute_import
from __future__ import print_function

import traceback
import random
import copy
import os
import json
import shutil

import tensorflow as tf

from nncompress.compression.lowrank import decompose
from nncompress.compression.pruning import prune, prune_filter
from nncompress.algorithms.solver.simulated_annealing import SimulatedAnnealingSolver
from nncompress.algorithms.solver.solver import State
from nncompress import backend as M

def random_sample(model, search_space, nsteps, use_same_spec=False, filter_func=None):
    actions = []
    nsteps_ = random.randint(1,nsteps+1)
    prev_targets = set()
    spec = None
    while len(actions) < nsteps_:
        if spec is None or not use_same_spec:
            spec = random.sample(search_space, 1)[0]
        kwargs = copy.deepcopy(spec[1])
        # Determine a method and its parameters
        for key in kwargs:
            if key in {"targets", "custom_objects"}:
                continue
            else:
                domain = spec[1][key]
                if type(domain) == tuple:
                    if type(domain[0]) == float:
                        val = random.random() * (domain[1] - domain[0]) + domain[0]
                    elif type(domain[0]) == int:
                        val = random.randint(domain[0], domain[1])
                    else:
                        raise NotImplementedError("Unsupported domain value.")
                elif type(domain) == list:
                    val = random.sample(domain, 1)[0]
                kwargs[key] = val

        # Determine targets and compression strengths.
        layers = {"Dense", "Conv2D"}
        domain = [layer.name for layer in model.layers if layer.__class__.__name__ in layers]
        domain = [layer for layer in domain if layer not in prev_targets]
        if filter_func is not None and spec[0].__name__ in filter_func:
            domain = filter_func[spec[0].__name__](model, domain, **kwargs)

        target = random.sample(domain, 1)[0]
        prev_targets.add(target)
        strength = random.random() * (spec[1]["targets"][1] - spec[1]["targets"][0])\
            + spec[1]["targets"][0]
        kwargs["targets"] = [(target, strength)]

        actions.append((spec[0], kwargs))
    return actions

class CompressionState(State):

    def __init__(self, name, model, ctx, ancestors=None, log=None):
        super(CompressionState, self).__init__()
        self._name = name
        self._model = model
        self._ancestors = ancestors or []
        self._ctx = ctx
        self._score = None
        self.log = log

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    @property
    def score(self):
        return self._score

    @property 
    def ancestors(self):
        return self._ancestors

    def get_next_impl(self):
        ret = None
        while ret is None:
            ancestors = [a for a in self._ancestors]
            cidx = random.randint(0, len(ancestors))
            if cidx == len(ancestors):
                ancestors.append(self)
            else:
                while len(ancestors)-1 > cidx:
                    ancestors.pop()
            candidate = ancestors[-1]
            actions = random_sample(candidate.model, self._ctx.search_space, self._ctx.nsteps, use_same_spec=True, filter_func=self._ctx.filter_func)
            model = M.copy_(candidate.model)
            log = []
            masking = []
            for a in actions:
                print(a)
                try:
                    ret = a[0](model, **a[1])
                    if len(ret) == 2:
                        model, replace_mappings = ret
                    elif len(ret) == 3:
                        model, replace_mappings, history = ret
                        masking.append((history, a))
                    log.append((replace_mappings, a))
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("A problem occurs with %s(%s)" % (str(a[0]), str(a[1])))
                    break
            ret = CompressionState(
                name=self._ctx.generate_state_name(),
                model=model,
                ctx=self._ctx,
                ancestors=ancestors[-1*self._ctx.h:],
                log=log)
            if self._ctx.compression_callbacks is not None:
                pid = ret.ancestors[-1].name
                cid = ret.name
                for c in self._ctx.compression_callbacks:
                    c(pid, cid, ret.ancestors[-1].model, ret.model, log, masking, self._ctx.data_holder)
        return ret

class NNCompress(object):

    def __init__(self, model, helper, dir_=os.getcwd(), max_iters=1000, h=3, nsteps=3, search_space=None, compression_callbacks=None, finetune_callback=None, custom_objects=None, solver_kwargs=None, filter_func=None, overwrite=False):
        self._model = model # original model, which will not be modified.
        self._masks = {} # to mask gradients
        self._states = []
        self._max_iters = max_iters
        self.helper = helper
        self.h = h
        self.nsteps = nsteps
        self._id_cnt = {"state":-1}
        self.compression_callbacks = compression_callbacks
        if filter_func is None:
            self.filter_func = {"prune":prune_filter}
        self._solver_kwargs = solver_kwargs
 
        self.data_holder = {}

        self._search_space = search_space
        if self._search_space is None:
            self._search_space = [
                (decompose, {
                    "targets":(0.0, 1.0),
                    "custom_objects":custom_objects
                }),
                (prune, {
                    "targets":(0.0, 1.0),
                    "mode":["channel", "weight"],
                    "method":["magnitude"],
                    "custom_objects":custom_objects
                })
            ]

        self._finetune_callback = finetune_callback
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        else:
            if overwrite:
                shutil.rmtree(dir_)
                os.mkdir(dir_)
            else:
                print("%s does exist, so will be terminated..." % dir_)
                import sys
                sys.exit(1)
        self._dir_ = dir_

    @property
    def search_space(self):
        return self._search_space

    def generate_state_name(self):
        self._id_cnt["state"] += 1
        return "state_%d" % self._id_cnt["state"]

    def get_dir(self):
        return self._dir_
       
    def compress(self):
        init_state = CompressionState(name=self.generate_state_name(), model=self._model, ctx=self)

        self.history = []
        self.last_score = -1
        def finetune_callback(state, i, transition):
            if len(self.history) >= 10:
                max_score = -1
                for state_ in self.history:
                    if state_.score is None:
                        score(state_)
                    max_score = max(max_score, state_.score)
                if self.last_score != -1 and max_score / self.last_score < 1.05:
                    self.helper.train(state.model)
                    max_score = score(state, force=True)
                self.last_score = max_score 
                self.history.clear()
            else:
                self.history.append(state)

        def dump_callback(state, i, transition):
            if transition:
                dumping_path = os.path.join(self.get_dir(), "trained_models")
                if not os.path.exists(dumping_path):
                    os.mkdir(dumping_path)
                basepath = os.path.join(dumping_path, "%s_%.4f" % (state.name, state.score))
                tf.keras.models.save_model(state.model, basepath+".h5")
                info = {"score":state.score, "idx":i, "transition":transition}
                with open(basepath+".json", "w") as f:
                    json.dump(info, f)

        def score(state, force=False):
            if state.score is None or force:
                score_ = self.helper.score(state.model)
                state._score =  score_
            else:
                score_ = state.score
            return score_

        solver = SimulatedAnnealingSolver(score, self._max_iters, **(self._solver_kwargs or {}))
        finetune_cbk = finetune_callback if self._finetune_callback is None else self._finetune_callback
        return solver.solve(init_state, callbacks=[dump_callback, finetune_cbk])
