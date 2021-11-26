import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import albumentations as albu
from .randwired_model import randwired_cifar, WeightedSum, randwired_cct
import tensorflow_addons as tfa
from .utils import WarmUpCosineDecayScheduler


height = 32
width = 32
input_shape = (height, width, 3) # network input
batch_size = 64

initial_lr = 6e-4
weight_decay = 6e-2

def model_builder(hp):
    model = randwired_cifar()
    sgd = tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)
    return model

def get_name():
    return "randcct"

def preprocess_func(img):
    return img

def get_custom_objects():
    return {"WeightedSum":WeightedSum}

def batch_preprocess_func(img):
    composition = albu.Compose([
        albu.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    return composition(image=img)['image']

def get_model(n_classes=100):
    model = randwired_cct(num_classes=n_classes)
    return model

def get_train_epochs():
    return 300

def get_warmup_epochs():
    return 10

def compile(model, run_eagerly=False):
    opt = tfa.optimizers.AdamW(learning_rate=initial_lr, weight_decay=weight_decay*initial_lr, epsilon=1e-8)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=0.0, axis=-1,
                name='categorical_crossentropy'
                )
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nsteps=0):
    #reducing learning rate on plateau
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    #return [rlrop]
    #return [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]
    assert nsteps > 0
    warmup_epochs = get_warmup_epochs()
    total_steps = get_train_epochs() * nsteps
    warmup_steps = warmup_epochs * nsteps

    lr_cbk = WarmUpCosineDecayScheduler(
                 learning_rate_base=initial_lr,
                 total_steps=total_steps,
                 min_lr=1e-5,
                 warmup_learning_rate=0.000001,
                 warmup_steps=warmup_steps,
                 weight_decay=weight_decay,
                 hold_base_rate_steps=0)
    return [lr_cbk]
