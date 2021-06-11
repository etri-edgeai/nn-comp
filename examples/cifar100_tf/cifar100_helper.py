from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from nncompress.run.helper import Helper
from nncompress.backend.tensorflow_.data.augmenting_generator import AugmentingGenerator, cutmix
from nncompress.backend.tensorflow_.utils import count_all_params

class CIFAR100Helper(Helper):

    def __init__(self, use_cutmix=False):

        self.batch_size = 32
        self.num_classes = 100
        self.epochs = 10
        self.data_augmentation = True
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0005, nesterov=True)

        # Load data
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
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
        
    def train(self, model, epochs=-1, callbacks=None):
        """Train the input model.

        """
        if epochs == -1:
            epochs = self.epochs
        model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
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
                  validation_data=*self.test_data,
                  shuffle=True,
                  callbacks=callbacks)

    def evaluate(self, model):
        """Evaluate the input model in terms of accuracy.

        """
        model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
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
