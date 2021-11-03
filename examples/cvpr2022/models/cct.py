import networkx as nx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import albumentations as albu

from .utils import WarmUpCosineDecayScheduler

positional_emb = True
projection_dim = 256

num_heads = 4
mlp_ratio = 2.0
transformer_units = [
    projection_dim*mlp_ratio,
    projection_dim
]
transformer_layers = 7
stochastic_depth_rate = 0.1

num_conv_layers = 1

height = 32
width = 32
input_shape = (height, width, 3) # network input
batch_size = 64
dropout = 0.0
l2_reg= 6e-2

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        return {
            "drop_prop":self.drop_prob
        }

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def seq_conv(num_conv_layers_, kernel_size=3, stride=1, padding=1, pooling_kernel_size=3, pooling_stride=2, num_output_channels=[256, 256]):
    conv_model = keras.Sequential()
    for i in range(num_conv_layers_):
        conv_model.add(
            layers.Conv2D(
                num_output_channels[i],
                kernel_size,
                stride,
                padding="valid",
                use_bias=True,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )
        #conv_model.add(layers.Dropout(dropout))
        conv_model.add(layers.ReLU())
        conv_model.add(layers.ZeroPadding2D(padding))
        conv_model.add(
            layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")
        )
    return conv_model

def conv_reshaping(conv_model, inputs):
    outputs = conv_model(inputs)
    print(outputs.shape)
    reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
    )
    return reshaped

def create_cct_model(
    num_classes=100,
    image_size=32,
    input_shape=None,
    input_tensor=None,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
    kernel_size=3):

    if input_tensor is not None:
        inputs = input_tensor
    else:
        inputs = layers.Input(input_shape)

    conv_model = seq_conv(num_conv_layers, kernel_size=kernel_size)

    # Encode patches.
    encoded_patches = conv_reshaping(conv_model, inputs)

    # Apply positional embedding.
    if positional_emb:
        seq_length = 16*16

        pos_embed = layers.Embedding(
            input_dim=seq_length, output_dim=projection_dim
        )

        #positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = np.arange(seq_length)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.0
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(l2_reg))(weighted_representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

    
def get_name():
    return "cct"

def preprocess_func(img):
    return img

def batch_preprocess_func(img):
    composition = albu.Compose([
        albu.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    return composition(image=img)['image']

def get_model(n_classes=100):
    return create_cct_model(num_classes=100, input_shape=input_shape)

def get_train_epochs():
    return 300

def get_warmup_epochs():
    return 10

initial_lr = 6e-4
def compile(model, run_eagerly=False):
    sgd = tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def lr_scheduler(epoch, lr):
    if epoch == 200:
        lr = lr * 0.1
    elif epoch == 250:
        lr = lr * 0.1
    print(lr)
    return lr

def get_callbacks(nsteps=0):
    #reducing learning rate on plateau
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    #return [rlrop]
    #return [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]
    assert nsteps > 0
    warmup_epochs = get_warmup_epochs()
    total_steps = (get_train_epochs()+warmup_epochs) * nsteps
    warmup_steps = warmup_epochs * nsteps

    lr_cbk = WarmUpCosineDecayScheduler(
                 learning_rate_base=initial_lr,
                 total_steps=total_steps,
                 warmup_learning_rate=0.000001,
                 warmup_steps=warmup_steps,
                 hold_base_rate_steps=0)
    return [lr_cbk]
