
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2

import efficientnet.tfkeras as efn

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 32

def get_name():
    return "densenet"

def preprocess_func(img):
    img = tf.keras.applications.densenet.preprocess_input(img)
    #img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    img = tf.image.resize(img, (height, width))
    return img

def get_model(n_classes=100):
    densenet = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
    densenet.trainable = True
    for layer in densenet.layers:
      if 'conv5' in layer.name:
        layer.trainable = True
      else:
        layer.trainable = False
    model = Sequential()
    model.add(densenet)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_initializer=keras.initializers.he_normal(seed=32)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.he_normal(seed=32)))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax', kernel_initializer=keras.initializers.he_normal(seed=32)))
    return model

def compile(model, run_eagerly=True):
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks():
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]
