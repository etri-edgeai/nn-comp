from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from nncompress.run.helper import Helper
from nncompress.backend.tensorflow_.data.augmenting_generator import AugmentingGenerator, cutmix
from nncompress.backend.tensorflow_.utils import count_all_params

class CIFAR100Helper(Helper):

    def __init__(self, num_classes=100, use_cutmix=False):

        self.batch_size = 32
        self.num_classes = num_classes
        self.epochs = 10
        self.data_augmentation = True

        # Load data
        if num_classes == 10:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif num_classes == 100:
            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        else:
            raise ValueError("num_classes invalid")

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

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

        self.datagen = datagen
        self.augmentation = cutmix if use_cutmix else None
        self.training_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

    def setup(self, model):
        self._original_params = count_all_params(model)
        
    def train(self, model, tag="none", init_lr=1e-3, epochs=-1, save_ckpt=False, callbacks=None):
        """Train the input model.

        """
        if epochs == -1:
            epochs = self.epochs

        def lr_schedule(epoch):
            """Learning Rate Schedule

            Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
            Called automatically every epoch as part of callbacks during training.

            # Arguments
                epoch (int): The number of epochs

            # Returns
                lr (float32): learning rate
            """
            lr = init_lr
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

        if save_ckpt:
            # Prepare model model saving directory.
            save_dir = os.path.join(os.getcwd(), 'saved_models_compressed')
            model_name = '%s.{epoch:03d}.h5' % tag
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            filepath = os.path.join(save_dir, model_name)

            # Prepare callbacks for model saving and for learning rate adjustment.
            checkpoint = ModelCheckpoint(filepath=filepath,
                                         monitor='val_accuracy',
                                         verbose=1,
                                         save_best_only=True)
 
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        if save_ckpt:
            callbacks = [checkpoint, lr_reducer, lr_scheduler]
        else:
            callbacks = [lr_reducer, lr_scheduler]

        if self.data_augmentation:
            model.fit(AugmentingGenerator(
                self.datagen.flow(*self.training_data, batch_size=self.batch_size), self.augmentation),
                epochs=epochs,
                validation_data=self.test_data,
                callbacks=callbacks,
                workers=1)
        else:
            model.fit(*self.training_data,
                  batch_size=self.batch_size,
                  epochs=epochs,
                  validation_data=self.test_data,
                  shuffle=True,
                  callbacks=callbacks)

    def evaluate(self, model):
        """Evaluate the input model in terms of accuracy.

        """
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        return model.evaluate(*self.test_data, verbose=1)[1]

    def sample_training_data(self, nsamples):
        data = [
            self.datagen.flow(*self.training_data, batch_size=self.batch_size).next()
            for _ in range(nsamples)
        ]
        return data

    def score(self, model):
        """Compute the score of the model in compression aspect.

        """
        total = count_all_params(model)
        if total > self._original_params:
            cscore = 0.0
        else:
            cscore = 1.0 - total / self._original_params
        acc = self.evaluate(model)
        if acc < 0.1:
            return 0.0
        return (acc + cscore) / 2.0
