
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

def get_shape(dataset):
    return (height, width, 3) # network input

def get_batch_size(dataset):
    return batch_size

def get_name():
    return "resnet50"

def preprocess_func(img, dim):
    img = tf.keras.applications.resnet.preprocess_input(img)
    #img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    img = tf.image.resize(img, (height, width))
    return img

def get_model(dataset, n_classes=100):
    densenet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, classes=n_classes)
    model = Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    if dataset == "cifar100":
        model.add(Dropout(0.5))
    else:
        model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def get_train_epochs():
    return 100

def compile(model, run_eagerly=True):
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nstep):
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]

def get_custom_objects():
    return None
