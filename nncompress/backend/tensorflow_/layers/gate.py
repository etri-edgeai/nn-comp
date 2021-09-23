from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from nncompress.assets.formula.gate import DifferentiableGateFormula, SimplePruningGateFormula

class DifferentiableGate(layers.Layer, DifferentiableGateFormula):

    def __init__(self,
                 ngates,
                 sparsity=0.5,
                 reg_weight=0.5,
                 grad_shape_func=None,
                 init_func=tf.random_uniform_initializer(0, 1.0),
                 **kwargs):
        super(DifferentiableGate, self).__init__(**kwargs)
        self.ngates = ngates
        self.grad_shaping = grad_shape_func
        self.sparsity = sparsity
        self.reg_weight = reg_weight
        self.init_func = init_func

    def build(self, input_shape):
        self.gates = self.add_weight(name='gates', 
                                     shape=(self.ngates,),
                                     initializer=self.init_func,
                                     trainable=True)
        super(DifferentiableGate, self).build(input_shape)

    def call(self, input):
        return self.compute(input, training=tf.keras.backend.learning_phase()), self.binary_selection()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ngates)

    def get_config(self):
        config = super(DifferentiableGate, self).get_config()
        config.update({
            "ngates":self.ngates,
            "grad_shape_func":self.grad_shaping,
            "sparsity":self.sparsity ,
            "reg_weight":self.reg_weight,
        })
        return config

class SimplePruningGate(layers.Layer, SimplePruningGateFormula):

    def __init__(self,
                 ngates,
                 **kwargs):
        super(SimplePruningGate, self).__init__(**kwargs)
        self.ngates = ngates

    def build(self, input_shape):
        self.gates = self.add_weight(name='gates',
                                     shape=(self.ngates,),
                                     initializer="ones",
                                     trainable=True)
        super(SimplePruningGate, self).build(input_shape)

    def call(self, input):
        return self.compute(input), self.binary_selection()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ngates)

    def get_config(self):
        config = super(SimplePruningGate, self).get_config()
        config.update({
            "ngates":self.ngates
        })
        return config
