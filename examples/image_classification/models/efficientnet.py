
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2
import horovod.tensorflow.keras as hvd

USE_EFNET = True
if USE_EFNET:
    from efficientnet.tfkeras import EfficientNetB0
    from efficientnet.tfkeras import preprocess_input
else:
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 16

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

def get_shape(dataset):
    return (height, width, 3) # network input

def get_batch_size(dataset):
    return batch_size

def get_name():
    return "efnet"

def preprocess_func(img, shape):

    img = tf.keras.applications.imagenet_utils.preprocess_input(
        img, data_format=None, mode='torch'
        )

    #img = preprocess_input(img)
    return img

def get_model(dataset, n_classes=100):
    if dataset == "imagenet2012":
        model = EfficientNetB0(weights='imagenet')
        return model
    else:
        efnb0 = EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=input_shape, classes=n_classes)

        model = Sequential()
        model.add(efnb0)
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


def compile(model, run_eagerly=True, lr=None, post_opt=None):

    if lr is None:
        lr = 0.001
    optimizer = Adam(lr=lr)
    if post_opt is not None:
        optimizer = post_opt(optimizer)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks(nsteps=0):
    #early stopping to monitor the validation loss and avoid overfitting
    #early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]

def get_custom_objects():
    return None

def get_train_epochs(finetune=False):
    if finetune:
        return 50
    else:
        return 100

def fix_mean_variance():
    return tf.convert_to_tensor([[[[0.485, 0.456, 0.406]]]]), tf.convert_to_tensor([[[[0.229, 0.224, 0.225]]]])

def get_heuristic_positions():

    return [
        "block1a_project_bn",
        "block2b_add",
        "block3b_add",
        "block4c_add",
        "block5c_add",
        "block6d_add",
        "top_activation"
    ]
