import numpy.random
import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.models import load_model
import dill
from memory import NormalMemory
gym_env_path = '/home/thai/eclipse-workspace/FightingICEv4.5'
# java_env_path='/home/thai/jdk1.8.0_271/bin/java'
import gym_fightingice

action_space_num = 56


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(512, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.d3 = tf.keras.layers.Dense(64, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space_num, activation=None)
        # x = np.random.normal(size=(6144, ))
        # x = np.expand_dims(np.array(x).transpose().flatten(), axis=0)
        # x = tf.convert_to_tensor(x)
        # _ = self.call(x)

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


observation_space_shape = (143)
experience = []
import json
import pandas as pd
from tqdm import tqdm
from agent import AgentWithNormalMemory, AgentWithPER
from datetime import datetime

if __name__ == '__main__':
    epsilon = 1.0
    agentoo7 = AgentWithPER(epsilon=epsilon)
    try:
        agentoo7.load_model()
        agentoo7.load_memory()
    except Exception as ex:
        print(ex)
        agentoo7 = AgentWithPER(epsilon=epsilon)
    steps = 400
    env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=gym_env_path, port=8888, freq_restart_java=4)
    for s in range(steps):
        done = False
        try:
            state, _, _, _ = env.reset(p2='MctsAi')
        except:
            state = env.reset(p2='MctsAi')
        total_reward = 0
        start_time = datetime.now()
        while not done:
            env.render()
            if len(state) == 4:
                state = state[0]
            action = agentoo7.act(state)
            next_state, reward, done, _ = env.step(action)
            agentoo7.update_mem(state, action, reward, next_state, done)
            agentoo7.train()
            state = next_state
            total_reward += reward

            if done:
                print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agentoo7.epsilon))
                print("time for an episode {}", datetime.now() - start_time)
                print('Save model')
                agentoo7.save_model()
                agentoo7.save_memory()

