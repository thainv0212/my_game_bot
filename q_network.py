import random

import numpy.random
import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.models import load_model

# env = gym.make("LunarLander-v2")

gym_env_path = '/home/thai/eclipse-workspace/FightingICEv4.5'
# java_env_path='/home/thai/jdk1.8.0_271/bin/java'
import gym_fightingice

action_space_num = 56


class DDDQN(tf.keras.Model):
    def __init__(self):
        super(DDDQN, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(64, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_space_num, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a


observation_space_shape = (96, 64)


class exp_replay():
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *[np.array(observation_space_shape).prod()]), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *[np.array(observation_space_shape).prod()]),
                                       dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = np.array(state).transpose().flatten()
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = np.array(next_state).transpose().flatten()
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size=64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class agent():
    def __init__(self, gamma=0.99, replace=100, lr=0.001, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = exp_replay()
        self.batch_size = 64
        self.q_net = DDDQN()
        # self.target_net = DDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        # self.target_net.compile(loss='mse', optimizer=opt)

    def act(self, state):
        if np.random.rand() <= self.epsilon or state is None:
            return np.random.choice([i for i in range(action_space_num)])

        else:
            try:
                actions = self.q_net.advantage(np.array([np.array(state).transpose().flatten()]))

                action = np.argmax(actions)
                print('get action', action)
                # print('return action', action)
            except Exception as ex:
                action = np.random.choice([i for i in range(action_space_num)])
                print('random action', action)
                print(ex)
            return action

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)

    # def update_target(self):
    #     self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return

        # if self.trainstep % self.replace == 0:
        #     self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        target = self.q_net.predict(states)
        # next_state_val = self.target_net.predict(next_states)
        next_state_val = self.q_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self):
        # self.q_net.save_weights("weights", save_format='tf')
        self.q_net.save('model.h5py', save_format='tf')
        # self.target_net.save("target_model.h5")

    def load_model(self):
        self.q_net = load_model("model.h5py")
        # self.q_net.load_weights("weights")
        # self.target_net = load_model("model.h5")


# agentoo7 = agent()
steps = 400
# for s in range(steps):
#     done = False
#     state = env.reset()
#     total_reward = 0
#     while not done:
#         env.render()
#         action = agentoo7.act(state)
#         next_state, reward, done, _ = env.step(action)
#         agentoo7.update_mem(state, action, reward, next_state, done)
#         agentoo7.train()
#         state = next_state
#         total_reward += reward
#
#         if done:
#             print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agentoo7.epsilon))
experience = []
import json


def create_data(epsilon=1):
    env = gym.make("FightingiceDisplayNoFrameskip-v0", java_env_path=gym_env_path, port=4242, freq_restart_java=3)
    obs = env.reset()
    low = env.observation_space.low
    high = env.observation_space.high
    agentoo7 = agent(epsilon=epsilon)
    agentoo7.load_model()
    while True:
        state, reward, done, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agentoo7.act(state)
            # action = numpy.random.randint(0, env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            if next_state is not None and state is not None:
                with open('train_data.txt', 'a') as f:
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(json.dumps(state.squeeze().tolist()), action, reward,
                                                          json.dumps(next_state.squeeze().tolist()), json.dumps(done)))
            # experience.append([state, action, reward, next_state, done])
            # agentoo7.update_mem(state, action, reward, next_state, done)
            # agentoo7.train()
            state = next_state
            total_reward += reward
            if done:
                print("total reward after is {}".format(total_reward))
    print('finish')


import pandas as pd
from tqdm import tqdm


def train():
    agentoo7 = agent()
    try:
        agentoo7.load_model()
    except Exception as ex:
        print(ex)
    with open('train_data.txt', 'r') as f:
        data = f.readlines()
        data = data[:200]
        for line in tqdm(data[::-1]):
            state, action, reward, next_state, done = line.split('\t')
            state = np.array(json.loads(state))
            done = json.loads(done)
            next_state = np.array(json.loads(next_state))
            agentoo7.update_mem(state, action, reward, next_state, done)
            agentoo7.train()
        agentoo7.save_model()


import sys
import os
if __name__ == '__main__':
    arg = sys.argv[1]
    if arg == 'run':
        epsilon = 0.3 if len(sys.argv) <= 2 else float(sys.argv[2])
        # try:
        #     os.remove('train_data.txt')
        # except:
        #     pass
        print('run with eploit rate', epsilon)
        create_data(epsilon)
    else:
        train()
# train()
# for i in range(5):
#     print('Loop {}'.format(i))
#     print('Training')
#     train()
#     print('Perform')
#     create_data()
# create_data(0.7)