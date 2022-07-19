
import tensorflow as tf
from tensorflow import keras

import horovod.tensorflow as hvd

from utils.optimizer_factory import GradientAccumulator

@tf.keras.utils.register_keras_serializable(package='Vision')
class GAModel(keras.Model):

    def __init__(self, use_amp, hvd_fp16_compression, grad_clip_norm, grad_accum_steps, *args, **kwargs):
        super(GAModel, self).__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.hvd_fp16_compression = hvd_fp16_compression
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.grad_accumulator = GradientAccumulator()
        self.gradients_gnorm = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.local_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf.function
    def train_step(self, data):
        """[summary]
        custom training step, which is used in case the user requests gradient accumulation.
        """

        # params
        use_amp = self.use_amp
        grad_accum_steps = self.grad_accum_steps
        hvd_fp16_compression = self.hvd_fp16_compression
        grad_clip_norm = self.grad_clip_norm
        
        #Forward and Backward pass
        x,y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)[0]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if use_amp:
                loss = self.optimizer.get_scaled_loss(loss)
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        #Backprop gradients
        # tape = hvd.DistributedGradientTape(tape, compression=hvd.Compression.fp16 if use_amp and hvd_fp16_compression else hvd.Compression.none)
        gradients = tape.gradient(loss, self.trainable_variables)

        #Get unscaled gradients if AMP
        if use_amp:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        #Accumulate gradients
        self.grad_accumulator(gradients)

        if self.local_step % grad_accum_steps == 0: 
            gradients = [None if g is None else hvd.allreduce(g / tf.cast(grad_accum_steps, g.dtype),
                                    compression=hvd.Compression.fp16 if use_amp and hvd_fp16_compression else hvd.Compression.none)
                                    for g in self.grad_accumulator.gradients]
            if grad_clip_norm > 0:
                (gradients, gradients_gnorm) = tf.clip_by_global_norm(gradients, clip_norm=grad_clip_norm)
                self.gradients_gnorm.assign(gradients_gnorm) # this will later appear on tensorboard
            #Weight update & iteration update
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.grad_accumulator.reset()
        

        # update local counter
        self.local_step.assign_add(1)
        
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
