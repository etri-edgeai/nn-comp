# coding: utf-8

from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
import efficientnet.tfkeras
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import os
import argparse

tf.random.set_seed(2)
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
ia.seed(1234)

import numpy as np
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
tf.executing_eagerly()
from tensorflow import keras

import albumentations as albu
from sklearn.model_selection import StratifiedShuffleSplit

# constants
n_classes = 100
epochs = 50
batch_size = 8

class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels=None, mode='fit', batch_size=batch_size, dim=(32, 32), channels=3, n_classes=n_classes, shuffle=True, augment=False, preprocess_func=None):
        
        #initializing the configuration of the generator
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        self.rand_aug = iaa.RandAugment(n=3, m=7)
        self.preprocess_func = preprocess_func
   
    #method to be called after every epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    #return numbers of steps in an epoch using samples and batch size
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    #this method is called with the batch number as an argument to obtain a given batch of data
    def __getitem__(self, index):
        #generate one batch of data
        #generate indexes of batch
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        
        #generate mini-batch of X
        X = np.empty((self.batch_size, *self.dim, self.channels))
        
        for i, ID in enumerate(batch_indexes):
            #generate pre-processed image
            img = self.images[ID]

            if self.preprocess_func is not None:
                img = self.preprocess_func(img)

            #resizing as per new dimensions
            X[i] = img
           
        #generate mini-batch of y
        if self.mode == 'fit':
            y = self.labels[batch_indexes]
            
            #augmentation on the training dataset
            if self.augment:
                X = self.__augment_batch(X)

            self.last = (X, y)
            return X, y
        
        elif self.mode == 'predict':
            return X
        
        else:
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")
            
    #augmentation for one image
    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                   albu.VerticalFlip(p=0.5),
                                   albu.GridDistortion(p=0.2),
                                   albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']
    
    #augmentation for batch of images
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
            #img_batch[i] = self.rand_aug(img_batch[i])
        return img_batch

def load_data(model_handler):

    dim = (model_handler.height, model_handler.width)
    preprocess_func = model_handler.preprocess_func

    if hasattr(model_handler, "batch_size"):
        batch_size_ = model_handler.batch_size
    else:
        batch_size_ = batch_size

    # Load the CIFAR10/100 data.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    num_classes = 100

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=123)

    for train_index, val_index in sss.split(x_train, y_train):
        x_train_data, x_val_data = x_train[train_index], x_train[val_index]
        y_train_data, y_val_data = y_train[train_index], y_train[val_index]

    print("Number of training samples: ", x_train_data.shape[0])
    print("Number of validation samples: ", x_val_data.shape[0])

    train_data_generator = DataGenerator(x_train_data, y_train_data, batch_size=batch_size_, augment=True, dim=dim, preprocess_func=preprocess_func)
    valid_data_generator = DataGenerator(x_val_data, y_val_data, batch_size=batch_size_, augment=False, dim=dim, preprocess_func=preprocess_func)
    test_data_generator = DataGenerator(x_test, y_test, batch_size=batch_size_, augment=False, dim=dim, preprocess_func=preprocess_func)
    
    return train_data_generator, valid_data_generator, test_data_generator


def train(model, model_name, model_handler, run_eagerly=False, callbacks=None):

    train_data_generator, valid_data_generator, test_data_generator = load_data(model_handler)

    callbacks = model_handler.get_callbacks()
    model_handler.compile(model)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name_ = '%s_model.{epoch:03d}.h5' % model_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_)

    mchk = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
    )

    model_history = model.fit_generator(train_data_generator,
                                    validation_data=valid_data_generator,
                                    callbacks=[mchk]+callbacks,
                                    verbose=1,
                                    epochs=epochs)

def prune(model, model_name):

    from group_fisher import make_group_fisher
    gmodel, pc = make_group_fisher(model, batch_size, target_ratio=0.1)
    optimizer = Adam(lr=0.0001)
    #gf.gmodel.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    train(gmodel, model_name+"_gmodel", run_eagerly=True, callbacks=[pc]) 



def run():
    parser = argparse.ArgumentParser(description='CIFAR100 ', add_help=False)
    parser.add_argument('--model_path', type=str, default=None, help='model')
    parser.add_argument('--model_name', type=str, default=None, help='model')
    parser.add_argument('--mode', type=str, default="test", help='model')
    args = parser.parse_args()

    if args.model_name == "efnet":
        from models import efficientnet as model_handler
    elif args.model_name == "vit":
        from models import vit as model_handler
    elif args.model_name == "densenet":
        from models import densenet as model_handler
    elif args.model_name == "mlp":
        from models import mlp as model_handler

    if args.mode == "test": 
        model = tf.keras.models.load_model(args.model_path)
        _, _, test_data_gen = load_data(model_handler)
        print(model.evaluate(test_data_gen, verbose=1)[1])
    elif args.mode == "train": # train
        model = model_handler.get_model(n_classes=n_classes)
        train(model, model_handler.get_name(), model_handler)
    elif args.mode == "prune":
        model = tf.keras.models.load_model(args.model_path)
        prune(model, model_handler) 

if __name__ == "__main__":
    run()