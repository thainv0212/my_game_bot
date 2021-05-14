import json
import math
import random

import numpy as np
from py4j.java_collections import JavaArray
import const
import dill

class State:
    data = []

    def __init__(self, frame_data, player, string_data=None):
        self.frame_data = frame_data
        self.player = player
        if frame_data is not None and player is not None:
            self.data = self.get_observation()
        if string_data is not None:
            self.from_str(string_data)
        # print('observation', self.data)

    def to_str(self):
        data = self.data
        if isinstance(data, np.ndarray):
            data = data.tolist()
        data_str = json.dumps(data)
        return data_str

    def from_str(self, string):
        data = json.loads(string)
        self.data = np.array(data)

    def get_observation(self):
        my = self.frame_data.getCharacter(self.player)
        opp = self.frame_data.getCharacter(not self.player)

        myHp = abs(my.getHp() / 500)
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2) / 960
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 15
        mySpeedY = my.getSpeedY() / 28
        myState = my.getAction().ordinal()

        oppHp = abs(opp.getHp() / 500)
        oppEnergy = opp.getEnergy() / 300
        oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = opp.getSpeedX() / 15
        oppSpeedY = opp.getSpeedY() / 28
        oppState = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        observation = []
        observation.append(myHp)
        # observation.append(myEnergy)
        # observation.append(myX)
        # observation.append(myY)
        if mySpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        # observation.append(abs(mySpeedX))
        if mySpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        deltaX = math.fabs(myX - oppX)
        deltaY = math.fabs(myY - oppY)
        deltaEnergy = math.fabs(myEnergy - oppEnergy)
        observation.append(deltaX)
        observation.append(deltaY)
        observation.append(deltaEnergy)
        # observation.append(abs(mySpeedY))
        # for i in range(56):
        #     if i == myState:
        #         observation.append(1)
        #     else:
        #         observation.append(0)

        observation.append(oppHp)
        # observation.append(oppEnergy)
        # observation.append(oppX)
        # observation.append(oppY)
        if oppSpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedX))
        if oppSpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedY))
        # for i in range(56):
        #     if i == oppState:
        #         observation.append(1)
        #     else:
        #         observation.append(0)
        observation.append(oppRemainingFrame)

        myProjectiles = self.frame_data.getProjectilesByP1()
        oppProjectiles = self.frame_data.getProjectilesByP2()

        if len(myProjectiles) == 2:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            myHitDamage = myProjectiles[1].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
        elif len(myProjectiles) == 1:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        if len(oppProjectiles) == 2:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
        elif len(oppProjectiles) == 1:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        # print(len(observation))  #141
        observation = [round(i, 1) for i in observation]
        return observation


import os
class QFunction:
    Q = {}
    actions_index = None

    def __init__(self, actions_index):
        self.Q = {}
        self.actions_index = actions_index

    def get_value(self, state: State, action):
        if state is None:
            return np.random.randint(0, len(self.actions_index))
        key = state.to_str()
        val = self.Q.get(key, None)
        if val is None:
            val = np.zeros_like(self.actions_index)
            self.Q[key] = val
            return 0
        try:
            return val[action]
        except Exception as ex:
            print(ex)
            raise ex

    def set_value(self, state: State, action, value):
        self.Q[state.to_str()][action] = value

    def get_best_action(self, state: State):
        key = state.to_str()
        val = self.Q.get(key, None)
        if val is None:
            val = np.zeros_like(self.actions_index)
            self.Q[key] = val
            print('random action')
            return np.random.randint(len(self.actions_index))
        return np.argmax(self.Q[state.to_str()])


class AbstractReward:
    def __init__(self, player):
        self.player = player

    def get_reward(self, frame_data):
        pass


class Reward(AbstractReward):
    old_frame = None

    def get_reward(self, frame_data):
        if self.old_frame is None:
            return 0
        old_my_character = self.old_frame.getCharacter(self.player)
        old_my_hp = old_my_character.getHp()
        old_op_character = self.old_frame.getCharacter(not self.player)
        old_op_hp = old_op_character.getHp()

        current_my_character = frame_data.getCharacter(self.player)
        current_my_hp = current_my_character.getHp()
        current_op_character = frame_data.getCharacter(not self.player)
        current_op_hp = current_op_character.getHp()

        my_damage = old_my_hp - current_my_hp
        op_damage = old_op_hp - current_op_hp

        return op_damage - my_damage

    def update_frame(self, frame_data):
        self.old_frame = frame_data


class MyQAgent:
    dump_file = 'dump.txt'

    def __init__(self, player, actions=None, alpha=0.9, gamma=0.75, train=True, simulator=None, gateway=None):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.train = train
        self.simulator = simulator
        self.gateway = gateway

        self.rewards = Reward(self.player)
        self.actions_index = [i for i in range(len(self.actions))]
        if os.path.exists('Q_function.pkl'):
            print('read from dump file')
            self.Q = dill.load(open('Q_function.pkl', 'rb'))
        else:
            print('init new Q function')
            self.Q = QFunction(self.actions_index)
        self.old_action = None
        self.old_frame_data = None
        self.old_state = None
        self.count = 0

    def train(self):
        pass

    def act(self, frame_data, train=False):
        state = State(frame_data, self.player)
        # if not train:
        #     action_idx = self.Q.get_best_action(state)
        # else:
        #     action_idx = np.random.choice(self.actions_index)
        #     # print('get random action', action_idx)
        # if train and self.old_action is not None:
        #     self.dump(frame_data)
        # temporal difference
        # todo perform action and get next state
        prob = 1
        if train:
            prob = random.random()
        if prob >= 0.6:
            action_idx = self.Q.get_best_action(state)
        else:
            action_idx = np.random.choice(self.actions_index)
        action = self.actions[action_idx]
        # object_class = self.gateway.jvm.java.lang.String
        # java_array = self.gateway.new_array(object_class, 1)
        # java_array[0] = action
        # next_state = self.simulator.simulate(frame_data, self.player, java_array, None, 2)
        if self.old_state is not None and train:
            # log data
            # old_reward = self.rewards.get_reward(self.old_frame_data)
            # self.dump(self.old_state, self.old_action, old_reward, state)
            TD = self.rewards.get_reward(frame_data) + \
                 self.gamma * self.Q.get_value(state, self.old_action) - \
                 self.Q.get_value(self.old_state,self.old_action) - 0.5
            tmp = self.Q.get_value(self.old_state, self.old_action)
            tmp += self.alpha * TD
            # print(self.rewards.get_reward(frame_data), self.gamma * self.Q.get_value(state, self.Q.get_best_action(state)) , tmp)
            self.Q.set_value(self.old_state, self.old_action, tmp)
            self.old_action = action_idx
            self.old_frame_data = frame_data
            self.old_state = state
            self.rewards.update_frame(frame_data)
            self.count += 1
            if self.count % 50 == 0:
                print('dump Q function to file')
                dill.dump(self.Q, open('Q_function.pkl', 'wb'))
        return action

    def update(self, frame_data):
        pass

    def dump(self, state, action, reward, next_state):
        with open(self.dump_file, 'a') as f:
            state_str = state.to_str()
            next_state_str = next_state.to_str()
            f.write('{}\t{}\t{}\n'.format(state_str, action, reward, next_state_str))

    def load_train_data(self):
        with open(self.dump_file, 'r') as f:
            all_data = f.readlines()
        train_data = []
        for line in all_data:
            dumps = line.split('\t')
            state = State(frame_data=None, player=None, string_data=dumps[0])
            action = dumps[1]
            reward = int(dumps[2])
            train_data.append([state, action, reward])


# MyQAgent(True, const.ACTIONS).load_train_data()
# todo: add train function
