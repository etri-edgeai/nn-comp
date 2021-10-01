
import networkx as nx

height = 224
width = 224
input_shape = (height, width, 3) # network input
batch_size = 8

def build_cifar100()
    pass

def build_imagenet()
    pass

def define_graph()

    # Normal Node

    # Stage Node

    # ...
    pass

def construct_model_from_graph()
    pass

def get_name():
    return "randwired"

def preprocess_func(img):
    img = tf.image.resize(img, (height, width))
    img = keras.applications.imagenet_utils.preprocess_input(img, mode='tf') # model="tf" or "torch"
    return img

def get_model(n_classes=100):
    # TODO
    return model

def compile(model, run_eagerly=False):
    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=run_eagerly)

def get_callbacks():
    #reducing learning rate on plateau
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 5, factor= 0.5, min_lr= 1e-6, verbose=1)
    return [rlrop]
