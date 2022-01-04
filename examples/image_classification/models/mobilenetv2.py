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
    return "mobilenetv2"

def preprocess_func(img, shape):
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = tf.image.resize(img, (height, width))
    return img

def get_model(dataset, n_classes=100):
    model_ = tf.keras.applications.mobilenet_v2.MobileNetV2(
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

def get_optimizer(mode=0):
    if mode == 0:
        return Adam(lr=0.0001)
    elif mode == 1:
        return Adam(lr=0.00001)

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

def get_heuristic_positions():

    return [
        "block_1_project_BN",
        "block_2_add",
        "block_3_project_BN",
        "block_4_add",
        "block_5_add",
        "block_6_project_BN",
        "block_7_add",
        "block_8_add",
        "block_9_add",
        "block_10_project_BN",
        "block_11_add",
        "block_12_add",
        "block_13_project_BN",
        "block_14_add",
        "block_15_add",
        "block_16_project_BN",
        "out_relu"
    ]
