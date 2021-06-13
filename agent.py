import dill
import numpy as np
import tensorflow as tf

from ddqn import DDDQN, PERDDDQN
from ddqn import advantage
from memory import NormalMemory, PERMemory

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model


class AgentWithNormalMemory():
    def __init__(self, gamma=0.1, replace=100, lr=0.01, epsilon=1.0, action_space_num=(56)):
        self.gamma = gamma
        self.epsilon = epsilon
        # self.min_epsilon = 0.4
        self.min_epsilon = 0.1
        self.epsilon_decay = 1e-5
        self.replace = replace
        self.trainstep = 0
        self.memory = NormalMemory()
        self.batch_size = 64
        self.q_net = DDDQN()
        self.target_net = DDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        # self.q_net.compile(loss='mse', optimizer=opt)
        # self.target_net.compile(loss='mse', optimizer=opt)
        self.q_net.compile(loss=tf.keras.losses.Huber(), optimizer=opt)
        self.target_net.compile(loss=tf.keras.losses.Huber(), optimizer=opt)
        self.action_space_num = action_space_num

    def act(self, state):
        # print('give action with epsilon {}'.format(self.epsilon))
        if np.random.rand() <= self.epsilon or state is None:
            return np.random.choice([i for i in range(self.action_space_num)])
        else:
            try:
                # actions = advantage(self.q_net, np.array([np.array(state).transpose().flatten()]))
                # test = self.q_net(np.array([np.array(state).transpose().flatten()]))
                # np_state = np.expand_dims(np.array(state).transpose(), axis=0)
                np_state = np.squeeze(state).transpose().flatten()
                np_state = np.expand_dims(np_state, axis=0)
                actions = advantage(self.q_net, np_state)
                action = np.argmax(actions)
                # print('action', action)
                # print('return action', action)
            except Exception as ex:
                action = np.random.choice([i for i in range(self.action_space_num)])
                raise ex
            return action

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        # states = np.expand_dims(np.array(state).transpose(), axis=0)
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)
        # next_state_val = self.q_net.predict(next_states)
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * (1 - dones)
        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self):
        # self.q_net.save_weights('duel_model', save_format='tf')
        # self.target_net.save_weights('duel_model_target', save_format='tf')
        # self.q_net.save_weights("weights", save_format='tf')
        self.q_net.save('duel_model')
        self.target_net.save('duel_model_target', save_format='tf')
        # self.target_net.save("target_model", save_format='tf')

    def load_model(self):
        # self.q_net.load_weights('duel_model')
        # self.target_net.load_weights('duel_model_target')
        tmp = load_model("duel_model")
        self.q_net = tmp
        tmp = load_model('duel_model_target')
        self.target_net = tmp
        # # self.q_net.load_weights("weights")
        # # self.target_net = load_model("model.h5")

    def save_memory(self):
        print('save memory')
        dill.dump(self.memory, open('memory.pkl', 'wb'))

    def load_memory(self):
        try:
            memory = dill.load(open('memory.pkl', 'rb'))
            self.memory = memory
            print('load memory')
        except Exception as ex:
            print(ex)

    def set_trainable(self, trainable):
        self.q_net.trainable = trainable
        self.target_net.trainable = trainable

class AgentWithPER(AgentWithNormalMemory):
    def __init__(self, gamma=0.1, replace=100, lr=0.01, epsilon=1.0, action_space_num=(56)):
        super().__init__()
        # super().__init__(gamma, replace, lr, epsilon, action_space_num)
        self.gamma = gamma
        self.epsilon = epsilon
        # self.min_epsilon = 0.4
        self.min_epsilon = 0.1
        self.epsilon_decay = 1e-5
        self.replace = replace
        self.trainstep = 0
        self.memory = PERMemory(capacity=1000)
        self.batch_size = 64
        self.q_net = PERDDDQN()
        self.target_net = PERDDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
        self.action_space_num = action_space_num
        try:
            self.load_model()
            self.load_memory()
        except:
            pass

    def train(self):
        # if self.memory.tree.data_pointer < self.batch_size:
        #     return

        # if self.trainstep % self.replace == 0:
        #     self.update_target()
        self.update_target()
        b_idx, experiences, is_weights = self.memory.sample(self.batch_size)
        states = np.array([each[0][0] for each in experiences])
        actions = np.array([each[0][1] for each in experiences])
        rewards = np.array([each[0][2] for each in experiences])
        next_states = np.array([each[0][3] for each in experiences])
        dones = np.array([each[0][4] for each in experiences])

        # states = np.expand_dims(np.array(state).transpose(), axis=0)
        is_weights = is_weights
        target = self.q_net.predict(states)
        weight_one = np.ones_like(is_weights)
        next_state_val = self.target_net.predict(next_states)
        # next_state_val = self.q_net.predict(next_states)
        absolute_errors = np.copy(tf.abs(self.q_net.predict(next_states) - self.target_net.predict(next_states)))
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * (1 - dones)
        loss = self.q_net.train_on_batch(states, q_target, is_weights)
        # print('loss on training', loss)
        self.update_epsilon()
        # update memory
        self.memory.batch_update(b_idx, absolute_errors.mean(axis=1))
        self.trainstep += 1

    def update_mem(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.memory.store(experience)
