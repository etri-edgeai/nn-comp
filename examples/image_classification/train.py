import math
import os

from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from datagen_ds import DataGenerator

# constants
epochs = 50

def load_data(dataset, model_handler, training_augment=True, batch_size=-1, n_classes=100):

    dim = (model_handler.get_shape(dataset)[0], model_handler.get_shape(dataset)[1])
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

    if dataset == "imagenet":
        import sys
        sys.path.insert(0, "/home/jongryul/work/keras_imagenet")
        from utils.dataset import get_dataset
        dataset_dir = "/ssd_data2/jongryul/tf"
        ds_train = get_dataset(dataset_dir, 'train', batch_size_)
        ds_val = get_dataset(dataset_dir, 'validation', batch_size_)
        train_examples = 1281167
        val_examples = 50000
        preprocess_func = None
        batch_pf = None
        augment = False
        is_batched = True

        train_data_generator = ds_train
        valid_data_generator = ds_val
        test_data_generator = ds_val
    else:
        ds_train = tfds.load(dataset, split="train").cache()
        ds_val = tfds.load(dataset, split="test").cache()
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

def train(dataset, model, model_name, model_handler, run_eagerly=False, callbacks=None, is_training=True, augment=True, exclude_val=False, dir_="saved_models", n_classes=100, save_dir=None):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, training_augment=augment, n_classes=n_classes)

    if is_training and hasattr(model_handler, "get_train_epochs"):
        epochs_ = model_handler.get_train_epochs()
    else:
        epochs_ = epochs

    if callbacks is None:   
        callbacks = []

    if dataset == "imagenet":
        iters = int(math.ceil(1281167.0 / model_handler.batch_size))
    else:
        iters = len(train_data_generator)

    callbacks_ = model_handler.get_callbacks(iters)
    model_handler.compile(model, run_eagerly=run_eagerly)

    print(model.count_params())

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
                                        callbacks=callbacks+callbacks_,
                                        verbose=1,
                                        epochs=epochs_,
                                        steps_per_epoch=iters)
    else:
        model_history = model.fit(train_data_generator,
                                        validation_data=valid_data_generator,
                                        callbacks=callbacks+callbacks_,
                                        verbose=1,
                                        epochs=epochs_,
                                        steps_per_epoch=iters)


def train_step(X, model, teacher_logits=None, y=None):

    if teacher_logits is not None:
        len_outputs = len(teacher_logits)
    else:
        len_outputs = 0

    with tf.GradientTape() as tape:
        logits = model(X)
        if type(logits) != list:
            logits = [logits]
        if len_outputs > 0:
            loss = None
            for i in range(len_outputs-1): # exclude the model's ouptut logit.
                if type(logits[i+1]) == list:
                    temp = None
                    for j in range(len(logits[i+1])):
                        if temp is None:
                            temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(logits[i+1][j], teacher_logits[i+1][j]))
                        else:
                            temp += tf.math.reduce_mean(tf.keras.losses.mean_squared_error(logits[i+1][j], teacher_logits[i+1][j]))
                    temp /= len(logits[i+1])
                else:
                    temp = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(logits[i+1], teacher_logits[i+1]))

                if loss is None:
                    loss = temp
                else:
                    loss += temp

            loss += tf.math.reduce_mean(tf.keras.losses.kl_divergence(logits[0], teacher_logits[0])) # model's output logit
            if y is not None:
                loss += tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(logits[0], y))
        else:
            assert y is not None
            loss = tf.math.reduce_mean(tf.keras.losses.categorical_crossentropy(logits[0], y))

    return tape, loss


def iteration_based_train(dataset, model, model_handler, epochs, teacher=None, with_label=True, with_distillation=True, callback_before_update=None, stopping_callback=None, augment=True, n_classes=100):

    train_data_generator, valid_data_generator, test_data_generator = load_data(dataset, model_handler, training_augment=augment, n_classes=n_classes)

    if dataset == "imagenet":
        iters = int(math.ceil(1281167.0 / model_handler.batch_size))
    else:
        iters = len(train_data_generator)

    global_step = 0
    callbacks_ = model_handler.get_callbacks(iters)
    optimizer = model_handler.get_optimizer()
    for epoch in range(epochs):
        done = False
        idx = 0
        for X, y in train_data_generator:
            idx += 1
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            if teacher is not None:
                teacher_logits = teacher(X)
                if type(teacher_logits) != list:
                    teacher_logits = [teacher_logits]
            else:
                teacher_logits = None

            if callback_before_update is not None:
                callback_before_update(idx, global_step, X, model, teacher_logits, y)

            if with_label:
                if with_distillation:
                    tape, loss = train_step(X, model, teacher_logits, y)
                else:
                    tape, loss = train_step(X, model, None, y)
            else:
                tape, loss = train_step(X, model, teacher_logits, None)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            global_step += 1
            if stopping_callback is not None and stopping_callback(idx, global_step):
                done = True
                break
        if done:
            break
        else:
            train_data_generator.on_epoch_end()
