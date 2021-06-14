import tensorflow as tf


class DDDQN(tf.keras.Model):
    def __init__(self, action_space_num=(56)):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(512, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.d3 = tf.keras.layers.Dense(64, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space_num, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        x = self.d3(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        x = self.d3(x)
        a = self.a(x)
        return a


def advantage(model, state):
    x = model.d1(state)
    x = model.d2(x)
    x = model.d3(x)
    a = model.a(x)
    return a

class PERDDDQN(tf.keras.Model):
    def __init__(self, action_space_num=(56)):
        super(PERDDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(1024, activation='relu')
        self.d3 = tf.keras.layers.Dense(1024, activation='relu')
        self.d4 = tf.keras.layers.Dense(1024, activation='relu')
        self.d5 = tf.keras.layers.Dense(1024, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space_num, activation=None)

    def call(self, input):
        x = self.d1(input)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q