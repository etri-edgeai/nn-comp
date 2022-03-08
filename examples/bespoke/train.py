import math
import os
import logging

from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from datagen_ds import DataGenerator

def load_data(dataset, model_handler, training_augment=True, batch_size=-1, n_classes=100):

    dim = (224, 224)
    preprocess_func = model_handler.preprocess_func
    if hasattr(model_handler, "batch_preprocess_func"):
        batch_pf = model_handler.batch_preprocess_func
    else:
        batch_pf = None

    if batch_size == -1:
        batch_size_ = model_handler.get_batch_size(dataset)
    else:
        batch_size_ = batch_size

    augment = True
    reg_augment = True

    if dataset == "imagenet2012":
        ds_train = tfds.load(dataset, split="train")
        ds_val = tfds.load(dataset, split="validation")
    else:
        ds_train = tfds.load(dataset, split="train")
        ds_val = tfds.load(dataset, split="test")
    train_examples = None
    val_examples = None
    is_batched = False

    train_data_generator = DataGenerator(
        ds_train,
        dataset=dataset,
        batch_size=batch_size_,
        augment=training_augment and augment,
        reg_augment=training_augment and reg_augment,
        dim=dim,
        n_classes=n_classes,
        n_examples=train_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf)

    valid_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf)

    test_data_generator = DataGenerator(
        ds_val,
        dataset=dataset,
        batch_size=batch_size_,
        augment=False,
        dim=dim,
        n_classes=n_classes,
        n_examples=val_examples,
        preprocess_func=preprocess_func,
        is_batched=is_batched,
        batch_preprocess_func=batch_pf)

    return train_data_generator, valid_data_generator, test_data_generator

def train(dataset, model, model_name, model_handler, epochs, callbacks=None, augment=True, exclude_val=False, n_classes=100, save_dir=None):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, training_augment=augment, n_classes=n_classes)

    if callbacks is None:   
        callbacks = []

    iters = len(train_data_generator)

    # Prepare model model saving directory.
    if save_dir is not None:
        model_name_ = '%s_model.{epoch:03d}.h5' % (model_name+"_"+dataset)
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
        callbacks.append(mchk)

    if exclude_val:
        model_history = model.fit(train_data_generator,
                                        callbacks=callbacks,
                                        verbose=1,
                                        epochs=epochs,
                                        steps_per_epoch=iters)
    else:
        model_history = model.fit(train_data_generator,
                                        validation_data=valid_data_generator,
                                        callbacks=callbacks,
                                        verbose=1,
                                        epochs=epochs,
                                        steps_per_epoch=iters)
