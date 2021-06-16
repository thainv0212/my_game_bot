import dill
import numpy as np
import tensorflow as tf

from ddqn import DDDQN, PERDDDQN
from ddqn import advantage
from memory import NormalMemory, PERMemory

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])


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
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
        self.action_space_num = action_space_num
        self.memory_file = 'memory.pkl'

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
        self.q_net.save('duel_model')
        self.target_net.save('duel_model_target', save_format='tf')

    def load_model(self):
        tmp = load_model("duel_model")
        self.q_net = tmp
        tmp = load_model('duel_model_target')
        self.target_net = tmp

    def save_memory(self):
        print('save memory')
        dill.dump(self.memory, open(self.memory_file, 'wb'))

    def load_memory(self):
        try:
            memory = dill.load(open(self.memory_file, 'rb'))
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
        self.memory = PERMemory(capacity=10000)
        self.batch_size = 64
        self.q_net = PERDDDQN()
        self.target_net = PERDDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
        self.action_space_num = action_space_num
        self.memory_file = 'memory_per.pkl'
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
        next_state_val = self.target_net.predict(next_states)
        absolute_errors = np.copy(np.abs(self.q_net.predict(next_states) - self.target_net.predict(next_states)))
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * (1 - dones)
        loss = self.q_net.train_on_batch(states, q_target, is_weights)
        self.update_epsilon()

        # update memory
        self.memory.batch_update(b_idx, absolute_errors.mean(axis=1))
        self.trainstep += 1

    def update_mem(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.memory.store(experience)


class AgentWithPERAndMultiRewards(AgentWithNormalMemory):
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
        self.memory = PERMemory(capacity=10000)
        self.batch_size = 64
        self.q_net_offensive = PERDDDQN()
        self.q_net_defensive = PERDDDQN()
        self.target_net_offensive = PERDDDQN()
        self.target_net_defensive = PERDDDQN()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net_offensive.compile(loss='mse', optimizer=opt)
        self.q_net_defensive.compile(loss='mse', optimizer=opt)
        self.target_net_offensive.compile(loss='mse', optimizer=opt)
        self.target_net_defensive.compile(loss='mse', optimizer=opt)
        self.action_space_num = action_space_num
        self.memory_file = 'memory_multi_per.pkl'
        try:
            self.load_model()
            self.load_memory()
        except:
            pass

    def save_model(self):
        self.q_net_offensive.save('q_net_offensive')
        self.q_net_defensive.save('q_net_defensive')
        self.target_net_offensive.save('target_net_offensive')
        self.target_net_defensive.save('target_net_defensive')

    def load_model(self):
        tmp = load_model("q_net_offensive")
        self.q_net_offensive = tmp
        tmp = load_model('q_net_defensive')
        self.q_net_defensive = tmp
        tmp = load_model("target_net_offensive")
        self.target_net_offensive = tmp
        tmp = load_model('target_net_defensive')
        self.target_net_defensive = tmp

    def update_target(self):
        self.target_net_offensive.set_weights(self.q_net_offensive.get_weights())
        self.target_net_defensive.set_weights(self.q_net_defensive.get_weights())

    def train(self):
        self.update_target()
        b_idx, experiences, is_weights = self.memory.sample(self.batch_size)
        states = np.array([each[0][0] for each in experiences])
        actions = np.array([each[0][1] for each in experiences])
        rewards = np.array([each[0][2] for each in experiences])
        next_states = np.array([each[0][3] for each in experiences])
        dones = np.array([each[0][4] for each in experiences])

        rewards_offensive = np.array([s[0] for s in rewards])
        rewards_defensive = np.array([s[1] for s in rewards])

        # offensive
        target_offensive = self.q_net_offensive.predict(states)
        next_states_val_offensive = self.target_net_offensive.predict(next_states)
        absolute_errors_offensive = np.copy(tf.abs(self.q_net_offensive.predict(next_states) - self.target_net_offensive.predict(next_states)))
        # defensive
        target_defensive = self.q_net_defensive.predict(states)
        next_states_val_defensive = self.target_net_defensive.predict(next_states)
        absolute_errors_defensive = np.copy(tf.abs(self.target_net_defensive.predict(next_states) - self.target_net_defensive.predict(next_states)))
        # get max action
        # next_states_val = next_states_val_defensive + next_states_val_offensive
        next_state_val_q_target = self.target_net_offensive.predict(next_states) + self.target_net_defensive.predict(next_states)
        max_action = np.argmax(next_state_val_q_target, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # train for offensive
        q_target_offensive = np.copy(target_offensive)
        q_target_offensive[batch_index, actions] = rewards_offensive + self.gamma * next_states_val_offensive[batch_index, max_action] * (1- dones)
        loss_offensive = self.q_net_offensive.train_on_batch(states, q_target_offensive, is_weights)

        # train for defensive
        q_target_defensive = np.copy(target_defensive)
        q_target_defensive[batch_index, actions] = rewards_defensive + self.gamma * next_states_val_defensive[batch_index, max_action] * (1- dones)
        loss_defensive = self.q_net_defensive.train_on_batch(states, q_target_defensive, is_weights)

        # update memory
        absolute_errors = absolute_errors_defensive + absolute_errors_offensive
        self.memory.batch_update(b_idx, absolute_errors.mean(axis=1))
        self.trainstep += 1

    def update_mem(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if isinstance(reward, int):
            print('type int')
        self.memory.store(experience)