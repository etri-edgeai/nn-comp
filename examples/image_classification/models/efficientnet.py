
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2

import efficientnet.tfkeras as efn
from efficientnet.keras import center_crop_and_resize, preprocess_input

def center_crop_and_resize(image, image_size, crop_padding=32, interpolation='bicubic'):
    shape = tf.shape(image)
    h = shape[0]
    w = shape[1]

    padded_center_crop_size = tf.cast((image_size / (image_size + crop_padding)) * tf.cast(tf.math.minimum(h, w), tf.float32), tf.int32)
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[offset_height:padded_center_crop_size + offset_height,
                       offset_width:padded_center_crop_size + offset_width]

    resized_image = tf.keras.preprocessing.image.smart_resize(image, [image_size, image_size], interpolation=interpolation)
    return resized_image

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 32

def get_shape(dataset):
    return (height, width, 3) # network input

def get_batch_size(dataset):
    return batch_size

def get_name():
    return "efnet"

def preprocess_func(img, shape):
    img = preprocess_input(img)
    return img

def get_model(dataset, n_classes=100):
    if dataset == "imagenet":
        efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=True, input_shape=input_shape, classes=n_classes)
        return efnb0
    else:
        efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=n_classes)
        model = Sequential()
        model.add(efnb0)
        model.add(GlobalAveragePooling2D())
        if dataset == "cifar100":
            model.add(Dropout(0.5))
        else:
            model.add(Dropout(0.25))
        model.add(Dense(n_classes, activation='softmax'))
        return model

def get_optimizer():
    return Adam(lr=0.0001)

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

def get_train_epochs(finetune=False):
    if finetune:
        return 50
    else:
        return 100
