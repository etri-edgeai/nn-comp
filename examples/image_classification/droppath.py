import tensorflow as tf

ESP = 1e-2

class DropPath(tf.keras.layers.Layer):
    """Drop Path also known as the Stochastic Depth layer.

    Refernece:
        - https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
        - github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return (x / (keep_prob+ESP)) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_prob": self.drop_path_prob})
        return config
