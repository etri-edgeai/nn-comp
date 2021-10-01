
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

def get_name():
    return "efnet"

def preprocess_func(img):
    img = img.astype(np.float32)/255.
    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
    return img

def get_model(n_classes=100):
    efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)
    model = Sequential()
    model.add(efnb0)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def compile(model, run_eagerly=False):
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks():
    #early stopping to monitor the validation loss and avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [early_stop, rlrop]
