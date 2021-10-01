import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2

from vit_keras import vit

height = 224
width = 224
input_shape = (height, width, 3) # network input

def get_name():
    return "vit"

def preprocess_func(img):
    #img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    img = tf.image.resize(img, (height, width))
    img = vit.preprocess_inputs(img)
    return img

def get_model(n_classes=100):
    model = vit.vit_l32(
        image_size=height,
        activation='sigmoid',
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=n_classes
    )
    return model

def compile(model, run_eagerly=False):
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks():
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]
