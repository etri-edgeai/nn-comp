from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from nncompress.assets.formula.gate import DifferentiableGateFormula
from callback import SparsityCallback

train = False
batch_size = 128
num_classes = 100
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar100_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = tf.keras.applications.ResNet50(include_top=True, weights=None, pooling=None, input_shape=(32,32,3), classes=100)
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    #prayer = DifferentiableGateFormula.instantiate(64, 0.25)
    #model.add(prayer)

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    input_layer = Input(batch_shape=model.layers[0].input_shape)
    prev = input_layer
    for layer in model.layers:
        layer._inbound_nodes = []
        prev = layer(prev)
    model = Model([input_layer], [prev])
    """

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if train:
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().

        model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=[SparsityCallback(model)],
                workers=1)


    # Save model and weights
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

from mlcorekit.nncompress.projection import extract_sample_features
from mlcorekit.nncompress.projection import least_square_projection

from cifar100_helper import CIFAR100Helper
helper = CIFAR100Helper()

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

"""
from mlcorekit.compression.lowrank import decompose
from mlcorekit.compression.pruning import prune
#model_ = decompose(model, [("dense", 0.2)])
#model.add_loss(lambda: prayer.get_sparsity_loss())
model_, replace_mappings, history = prune(model, [("conv2d_3", 0.05)], mode="channel")

data = extract_sample_features(model, [model.get_layer("conv2d_3")], helper)
least_square_projection(model_, data, masking)

model_.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
scores = model_.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
"""

from cifar100_helper import CIFAR100Helper
from mlcorekit.nncompress.nncompress import NNCompress

helper = CIFAR100Helper()
helper.setup(model)

def build_projection(pid, cid, model, compressed, log, masking, dataholder):
    merged_masking = {}
    for masking_, a in reversed(masking):
        if "mode" in a[1] and a[1]["mode"] != "channel":
            continue
        for layer_name in masking_:
            assert masking_[layer_name] is not None
            if layer_name not in merged_masking: # leaf
                merged_masking[layer_name] = masking_[layer_name]
            else:
                input_mask = None
                output_mask = None
                if masking_[layer_name][0] is not None:
                    indices = np.where(masking_[layer_name][0])
                    input_mask = np.copy(masking_[layer_name][0])
                    input_mask[indices] = merged_masking[layer_name][0]
                if masking_[layer_name][1] is  not None:
                    indices = np.where(masking_[layer_name][1])
                    output_mask = np.copy(masking_[layer_name][1])
                    output_mask[indices] = merged_masking[layer_name][1]
                merged_masking[layer_name] = (input_mask, output_mask)

    layers = []
    for layer_name in merged_masking:
        if pid in dataholder and "feat_data" in dataholder[pid] and layer_name in dataholder[pid]["feat_data"]:
            continue
        try:
            layers.append(model.get_layer(layer_name))
        except ValueError: # ignore the case `layer_name` is included in the model.
            continue

    # Layer filtering
    layers_ = []
    for layer in layers:
        if layer.__class__.__name__ == "Conv2D" and layer.kernel_size == (1,1) and layer.strides==(1,1):
            layers_.append(layer)
        elif layer.__class__.__name__ == "Dense":
            layers_.append(layer)
    if len(layers_) == 0:
        return None

    temp_data = extract_sample_features(model, layers_, helper)
    for layer_name, feat_data in temp_data.items():
        if pid not in dataholder:
            dataholder[pid] = {}
        if "feat_data" not in dataholder[pid]:
            dataholder[pid]["feat_data"] = {}
        dataholder[pid]["feat_data"][layer_name] = feat_data
    if cid not in dataholder:
        dataholder[cid] = {}
    dataholder[cid]["merged_masking"] = merged_masking
 
def apply_projection(pid, cid, model, compressed, log, masking, dataholder):
    if pid not in dataholder or "feat_data" not in dataholder[pid]:
        return
    if cid not in dataholder or "merged_masking" not in dataholder[cid]:
        return
    feat_data = dataholder[pid]["feat_data"]
    least_square_projection(compressed, feat_data, dataholder[cid]["merged_masking"])

import random
random.seed(1234)

nncompress = NNCompress(model, helper, compression_callbacks=[build_projection, apply_projection])
x = nncompress.compress()
