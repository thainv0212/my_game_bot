import pickle
import numpy as np
# def my_relu(x):
#     return (x > 0) * x

def my_relu(x):
    return np.maximum(x, np.zeros(x.shape))

def my_linear(x):
    return x

class MyDenseLayer:
    weight = None
    bias = None
    activation = None

    def my_relu(self, x):
        return (x > 0) * x

    def my_linear(self, x):
        return x
    
    def my_leaky_relu(self, x):
        return np.maximum(x, x * 0.1)


    def __init__(self, layer=None):
        self.activation_remap = {
            'relu': self.my_relu,
            'linear': self.my_linear,
            'leaky_relu': self.my_leaky_relu
        }
        if layer is not None:
            self.weight = layer.weights[0].numpy()
            self.bias = layer.bias.numpy()
            if layer.activation is not None:
                self.activation = self.activation_functions[layer.activation]

    def __call__(self, x):
        val = x.dot(self.weight) + self.bias
        if self.activation is not None:
            val = self.activation_remap[self.activation](val)
        return val

    def to_json(self):
        return {
            'weight': self.weight,
            'bias': self.bias,
            'activation': self.activation
        }

    @staticmethod
    def from_json(data):
        layer = MyDenseLayer()
        layer.weight = data['weight']
        layer.bias = data['bias']
        layer.activation = data['activation']
        return layer
class MyPyNetwork:
    layers = []

    def from_network(self, network):
        layers = network.layers
        for layer in layers:
            converted_layer = MyDenseLayer(layer)
            self.layers.append(converted_layer)

    def __call__(self, x):
        tmp = x
        for layer in self.layers[:-2]:
            tmp = layer(tmp)
        tmp = self.layers[-1](tmp)
        return tmp

    def dump(self, path):
        layers_data = []
        for layer in self.layers:
            layers_data.append(layer.to_json())
        pickle.dump(layers_data, open(path, 'wb'))

    def dump_pkl(self, path):
        pickle.dump(self.layers, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        # self.layers = dill.load(open(path, 'rb'))
        json_data = pickle.load(open(path, 'rb'))
        self.layers = []
        for d in json_data:
            self.layers.append(MyDenseLayer.from_json(d))
            # self.layers.append(MyDenseLayer.from_json(d))