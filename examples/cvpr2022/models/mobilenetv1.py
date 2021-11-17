import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2

import efficientnet.tfkeras as efn

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 32

def get_shape(dataset):
    return (height, width, 3) # network input

def get_batch_size(dataset):
    return batch_size

def get_name():
    return "mobilenet"

def preprocess_func(img, shape):
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = tf.image.resize(img, (height, width))
    return img

def get_model(dataset, n_classes=100):
    model_ = tf.keras.applications.mobilenet.MobileNet(
         alpha=1.0, include_top=False, weights='imagenet',
            classes=n_classes
                )
    model = Sequential()
    model.add(model_)
    model.add(GlobalAveragePooling2D())
    if dataset == "cifar100":
        model.add(Dropout(0.5))
    else:
        model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def compile(model, run_eagerly=False):
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nsteps=0):
    #early stopping to monitor the validation loss and avoid overfitting
    #early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]

def get_custom_objects():
    return None

def get_train_epochs():
    return 100
