from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from nncompress.utils.mlck import get_saved_model_path
from examples.cifar100_tf.cifar100_helper import CIFAR100Helper

MODELS = {
    "seq":None,
    "resnet":None,
    "densenet":None,
    "mobilenet":None,
    "json":None
}
helper = CIFAR100Helper()

def get_seq_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('softmax'))

    input_layer = Input(batch_shape=model.layers[0].input_shape)
    prev = input_layer
    for layer in model.layers:
        layer._inbound_nodes = []
        prev = layer(prev)
    model = Model([input_layer], [prev])
    return model

def request_model(model_key):
    assert model_key in MODELS
    umodel_path = os.path.join(get_saved_model_path(), "unittest")
    if not os.path.exists(umodel_path):
        os.mkdir(umodel_path)

    if MODELS[model_key] is None:
        model_path = os.path.join(umodel_path, model_key +".h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            if model_key == "resnet":
                model = tf.keras.applications.ResNet50(include_top=True, input_shape=(32, 32, 3), weights=None, pooling=None, classes=100)
            elif model_key == "densenet":
                model = tf.keras.applications.DenseNet121(include_top=True, input_shape=(32, 32, 3), weights=None, pooling=None, classes=100)
            elif model_key == "mobilenetv2":
                model = tf.keras.applications.MobileNetV2(include_top=True, input_shape=(32, 32, 3), weights=None, pooling=None, classes=100)
            elif model_key == "json": # ResNet50 json TF 2.4.1
                with open("tests/model.json", "r") as json_file:
                    model = keras.models.model_from_json(json_file.read())
            elif model_key == "seq":
                model = get_seq_model()
            else:
                raise NotImplementedError("Not supported model. %s" % model_key)
            helper.train(model)
            tf.keras.models.save_model(model, model_path)
            print('Saved trained model at %s ' % model_path)

    compile_model(model)
    return model

def compile_model(model):
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

if __name__ == "__main__":

    model = request_model("resnet")
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
