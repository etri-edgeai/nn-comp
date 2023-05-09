import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='ortho')
class OrthoRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor=0.01, mode="rows"):
        self.factor = factor
        self.mode = mode

    def __call__(self, inputs):
        
        inputs = tf.squeeze(inputs)

        if self.mode == "rows":
            inputs = tf.math.l2_normalize(inputs, axis=1)
            product = tf.matmul(inputs, tf.transpose(inputs))
            size = inputs.shape[0]
        else:
            inputs = tf.math.l2_normalize(inputs, axis=0)
            product = tf.matmul(tf.transpose(inputs), inputs)
            size = inputs.shape[1]
        product_no_diagonal = product * (1.0 - tf.eye(size, dtype=inputs.dtype))
        num_pairs = size * (size - 1.0) / 2.0
        return (
            self.factor
            * 0.5
            * tf.reduce_sum(tf.abs(product_no_diagonal))
            / num_pairs
        )

    def get_config(self):
        return {"factor": float(self.factor), "mode": self.mode}    

