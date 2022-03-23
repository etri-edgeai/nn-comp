from nncompress.bespoke.base.structure import ModelHouse

import tensorflow as tf
import numpy as np
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import resnet50 as model_handler
from train import train

dataset = "cifar100"


#model = tf.keras.applications.ResNet50(include_top=False, weights=None, pooling=None, classes=10)
#model = tf.keras.applications.DenseNet121(include_top=False, weights=None, pooling=None, classes=10)

model = model_handler.get_model("cifar100")

tf.keras.utils.plot_model(model, to_file="original.png", show_shapes=True)

mh = ModelHouse(model)

house = mh.make_train_graph(range_=[3,5])

data = np.random.rand(1,224,224,3)
y = house(data)

tf.keras.utils.plot_model(house, to_file="house.pdf", show_shapes=True)

model_handler.compile(house, run_eagerly=True)

print(model.summary())
print(house.input)

train(dataset, house, "test", model_handler, 5, callbacks=None, augment=True, exclude_val=False, n_classes=100)

#mh.extract({5: 0, 1: 1})
