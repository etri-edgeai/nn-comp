import tensorflow as tf

class SparsityCallback(tf.keras.callbacks.Callback):

    def __init__(self, model):
        self.set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        print("\n------------------------")
        pruning_layers = []
        for layer in self.model.layers:
            if type(layer).__name__ == "DifferentiableGate":
                pruning_layers.append(layer)

        for layer in pruning_layers:
            print(float(layer.get_sparsity()))
        print("\n")
