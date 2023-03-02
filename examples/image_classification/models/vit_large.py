
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization
import numpy as np
import cv2
import horovod.tensorflow.keras as hvd

from keras_cv import models
from keras_cv.models.vit import transformer_encoder

from nncompress.backend.tensorflow_.transformation.pruning_parser import PruningNNParser, NNParser



height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 32

def unfold(model):

    cnt = 0
    prev = None
    block_input = None
    head = []
    for idx, layer in enumerate(model.layers):
        if type(layer).__name__ == "TransformerEncoder":
            c = layer.get_config()
            if cnt == 0:
                block_input = prev.output
            cnt += 1
            prev = layer
        else:
            if type(prev).__name__ == "TransformerEncoder":
                last_idx = idx
                break
            prev = layer
  
    head = model.layers[last_idx:]

    input_ = block_input
    for _ in range(cnt):
        input_ = transformer_encoder(
            input_, c["project_dim"], c["mlp_dim"], c["num_heads"], c["mlp_dropout"], c["attention_dropout"], activation=c["activation"], layer_norm_epsilon=c["layer_norm_epsilon"])
   
    x = input_
    for layer in head:
        x = layer(x)

    new_model = tf.keras.models.Model(inputs=model.input, outputs=x)

    parser = PruningNNParser(new_model)
    parser.parse()

    # weight transfer
    is_first = True
    layer_map = {}
    id2layer = {}
    for layer in new_model.layers:
        id2layer[parser.torder[layer.name]] = layer

    for layer in new_model.layers:
        try:
            model.get_layer(layer.name)
            has_layer = True
        except Exception as e:
            has_layer = False

        if has_layer:
            layer.set_weights(model.get_layer(layer.name).get_weights())
        else:
            starting_idx = parser.torder[layer.name]
            break
    
    cnt = 0
    for layer in model.layers:
        if type(layer).__name__ == "TransformerEncoder":
            id_ = cnt * 11 + starting_idx

            mh = id2layer[id_ + 1]
            assert type(mh).__name__ == "MultiHeadAttention"

            dense1 = id2layer[id_+ 5]
            assert type(dense1).__name__ == "Dense"

            dense2 = id2layer[id_+ 8]
            assert type(dense2).__name__ == "Dense"

            layer_norm1 = id2layer[id_]
            assert type(layer_norm1).__name__ == "LayerNormalization"

            layer_norm2 = id2layer[id_+4]
            assert type(layer_norm2).__name__ == "LayerNormalization"

            mh.set_weights(layer.attn.get_weights())
            dense1.set_weights(layer.dense1.get_weights())
            dense2.set_weights(layer.dense2.get_weights())
            layer_norm1.set_weights(layer.layer_norm1.get_weights())
            layer_norm2.set_weights(layer.layer_norm2.get_weights())

            cnt += 1

    x = np.random.rand(1, model.input.shape[1], model.input.shape[2], model.input.shape[3])
    assert np.sum(model(x)) == np.sum(new_model(x))


    return new_model

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
    return "vit"

def data_preprocess_func(img, shape):
    #img = center_crop_and_resize(img, height)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    return img
    #img = preprocess_input(img)

def model_preprocess_func(img, shape):
    """
    img = tf.keras.applications.imagenet_utils.preprocess_input(
        img, data_format=None, mode='torch'
        )
    """
    #img = preprocess_input(img)
    return img

def get_model(dataset, n_classes=100):

    if dataset == "imagenet2012":
        model = models.__dict__["vit"].ViTS32(include_rescaling=True, include_top=True, weights="imagenet", classes=1000, input_shape=(224,224,3))
        #model = models.__dict__["vit"].ViTTiny16(include_rescaling=True, include_top=True, weights="imagenet", classes=1000, input_shape=(224,224,3))

        model = unfold(model)    
    
        return model
    else:
        model = models.__dict__["vit"].ViTS32(include_rescaling=False, include_top=False, weights="imagenet", classes=n_classes, input_shape=(224,224,3))

        model = unfold(model)

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
        return Adam(lr=0.0001*hvd.size())
    elif mode == 1:
        return Adam(lr=0.00001*hvd.size())


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
