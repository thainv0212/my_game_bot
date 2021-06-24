import dill
import numpy as np
import pickle
import types
import inspect
from train_ddqn import AgentWithNormalMemory, AgentNormalMultiReward, AgentWithPER, AgentWithPERAndMultiRewards
from tensorflow.python.keras.activations import relu, linear
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from argparse import ArgumentParser
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

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
        self.activation_functions = {
            relu: 'relu',
            linear: 'linear',
            LeakyReLU: 'leaky_relu'
        }
        self.activation_remap = {
            'relu': self.my_relu,
            'linear': self.my_linear,
            'leaky_relu': self.my_leaky_relu
        }
        if layer is not None:
            self.weight = layer.weights[0].numpy()
            self.bias = layer.bias.numpy()
            if layer.activation is not None:
                if isinstance(layer.activation, types.FunctionType):
                    self.activation = self.activation_functions[layer.activation]
                elif inspect.isclass(type(layer.activation)):
                    self.activation = self.activation_functions[type(layer.activation)]

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
    def __init__(self):
        self.layers = []

    def from_network(self, network):
        layers = network.layers
        print('from network', len(network.layers), len(self.layers))
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


if __name__ == '__main__':
    agents = {
        'normal': AgentWithNormalMemory,
        'per': AgentWithPER,
        'per_multi': AgentWithPERAndMultiRewards,
        'normal_multi': AgentNormalMultiReward,
    }
    multi_reward_types = {
        'normal': False,
        'per': False,
        'per_multi': True,
        'normal_multi': True
    }
    parser = ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['normal', 'per', 'per_multi', 'normal_multi'], required=True)
    args = parser.parse_args()
    Agent = agents[args.agent]
    print('start converting model for ', Agent)
    my_agent = Agent()
    my_agent.load_model()
    multi_reward = multi_reward_types[args.agent]
    print('convert to numpy weights')
    if multi_reward:
        offensive_network = MyPyNetwork()
        offensive_network.from_network(my_agent.q_net_offensive)
        offensive_network.dump('q_offensive.pickle')
        print('q_offensive', len(offensive_network.layers))
        defensive_network = MyPyNetwork()
        defensive_network.from_network(my_agent.q_net_defensive)
        defensive_network.dump('q_defensive.pickle')
        print('q_defensive', len(defensive_network.layers), len(my_agent.q_net_defensive.layers))
    else:
        my_network = MyPyNetwork()
        my_network.from_network(my_agent.q_net)
        my_network.dump('my_network.pickle')
        print('q model')
    # my_network.dump_pkl('my_network.pickle')
    # my_network
    print('conversion done')
