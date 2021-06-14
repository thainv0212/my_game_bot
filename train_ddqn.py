# PYTHON_ARGCOMPLETE_OK
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
import argparse
action_space_num = 56

observation_space_shape = (143)
experience = []
import json
import pandas as pd
from tqdm import tqdm
from agent import AgentWithNormalMemory, AgentWithPER
from datetime import datetime
import argcomplete

def train_with_agent(agent, epsilon):
    agentoo7 = agent(epsilon=epsilon)
    # try:
    #     agentoo7.load_model()
    #     agentoo7.load_memory()
    # except Exception as ex:
    #     print(ex)
    #     agentoo7 = agent(epsilon=epsilon)
    steps = 400
    env_name = 'FightingiceDataNoFrameskip-v0'
    # env_name = 'FightingiceDataFrameskip-v0'
    env = gym.make(env_name, java_env_path=gym_env_path, freq_restart_java=100)
    for s in range(steps):
        done = False
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
if __name__ == '__main__':
    # epsilon = 1.0
    # agent = AgentWithPER
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--agent', type=str, choices=['normal', 'per'], default='per')
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    epsilon = args.epsilon
    agents = {
        'normal': AgentWithNormalMemory,
        'per': AgentWithPER,
    }
    agent = agents[args.agent]
    train_with_agent(agent, epsilon)

