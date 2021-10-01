import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2

from keras_cv_attention_models import mlp_family

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 8

def get_name():
    return "mlp"

def preprocess_func(img):
    img = tf.image.resize(img, (height, width))
    img = keras.applications.imagenet_utils.preprocess_input(img, mode='tf') # model="tf" or "torch"
    return img

def get_model(n_classes=100):
    #model = mlp_family.ResMLP_B24(num_classes=n_classes, pretrained="imagenet")
    model = mlp_family.MLPMixerB16(num_classes=n_classes, pretrained="imagenet")
    print(model.summary())
    return model

def compile(model, run_eagerly=False):
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks():
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]
