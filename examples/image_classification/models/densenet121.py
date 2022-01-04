
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
    return "densenet121"

def preprocess_func(img, dim):
    img = tf.keras.applications.densenet.preprocess_input(img)
    #img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    img = tf.image.resize(img, (height, width))
    return img

def get_model(dataset, n_classes=100):
    densenet = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape, classes=n_classes)
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

def get_optimizer(mode=0):
    if mode == 0:
        return Adam(lr=0.0001)
    elif mode == 1:
        return Adam(lr=0.00001)

def compile(model, run_eagerly=True):
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nstep):
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]

def get_custom_objects():
    return None

def get_heuristic_positions():

    return [
        "conv2_block1_concat",
        "conv2_block2_concat",
        "conv2_block3_concat",
        "conv2_block4_concat",
        "conv2_block5_concat",
        "conv2_block6_concat",
        "conv3_block1_concat",
        "conv3_block2_concat",
        "conv3_block3_concat",
        "conv3_block4_concat",
        "conv3_block5_concat",
        "conv3_block6_concat",
        "conv3_block7_concat",
        "conv3_block8_concat",
        "conv3_block9_concat",
        "conv3_block10_concat",
        "conv3_block11_concat",
        "conv3_block12_concat",
        "conv4_block1_concat",
        "conv4_block2_concat",
        "conv4_block3_concat",
        "conv4_block4_concat",
        "conv4_block5_concat",
        "conv4_block6_concat",
        "conv4_block7_concat",
        "conv4_block8_concat",
        "conv4_block9_concat",
        "conv4_block10_concat",
        "conv4_block11_concat",
        "conv4_block12_concat",
        "conv4_block13_concat",
        "conv4_block14_concat",
        "conv4_block15_concat",
        "conv4_block16_concat",
        "conv4_block17_concat",
        "conv4_block18_concat",
        "conv4_block19_concat",
        "conv4_block20_concat",
        "conv4_block21_concat",
        "conv4_block22_concat",
        "conv4_block23_concat",
        "conv4_block24_concat",
        "conv5_block1_concat",
        "conv5_block2_concat",
        "conv5_block3_concat",
        "conv5_block4_concat",
        "conv5_block5_concat",
        "conv5_block6_concat",
        "conv5_block7_concat",
        "conv5_block8_concat",
        "conv5_block9_concat",
        "conv5_block10_concat",
        "conv5_block11_concat",
        "conv5_block12_concat",
        "conv5_block13_concat",
        "conv5_block14_concat",
        "conv5_block15_concat",
        "relu"
    ]
