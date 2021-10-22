import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import albumentations as albu
from .randwired_model import randwired_cifar

height = 32
width = 32
input_shape = (height, width, 3) # network input
batch_size = 100

def get_name():
    return "randwired"

def preprocess_func(img):
    return img

def batch_preprocess_func(img):
    composition = albu.Compose([
        albu.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    return composition(image=img)['image']

def get_model(n_classes=100):
    model = randwired_cifar(100)
    return model

def get_train_epochs():
    return 100

initial_lr = 0.1
def compile(model, run_eagerly=False):
    sgd = tf.keras.optimizers.SGD(lr=initial_lr, decay=0.0, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def lr_scheduler(epoch, lr):
    if epoch == 50:
        lr = lr * 0.1
    elif epoch == 75:
        lr = lr * 0.1
    print(lr)
    return lr

def get_callbacks():
    #reducing learning rate on plateau
    #rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    #return [rlrop]
    return [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]
