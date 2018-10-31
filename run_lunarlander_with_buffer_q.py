#!/usr/bin/env python3
import copy
import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.optim as optim

from networks import LunarLanderNetwork
from agents import NNBufferQAgent
from replay import UniformReplayBuffer
from common import get_running_mean
from envs import get_lunar_lander


# For reproducibility
np.random.seed(0xc0ffee)
torch.backends.cudnn.deterministic = True
torch.manual_seed(0xc0ffee)
torch.cuda.manual_seed(0xc0ffee)

start_q_network = LunarLanderNetwork(100)

for replay_buffer_size in [100, 1000, 10000, 100000, 1000000]:

    # Setup Environment
    env = get_lunar_lander(seed=0xc0ffee)

    # Setup Agent
    q_network = copy.deepcopy(start_q_network)
    optimizer = optim.RMSprop(q_network.parameters(), lr=5e-4)
    replay_buffer = UniformReplayBuffer(replay_buffer_size)
    agent = NNBufferQAgent(q_network, optimizer, replay_buffer, env.observation_space, env.action_space, epsilon=0.1, gamma=1, batch_size=10)

    # Train agent
    NB_EPISODES = 500
    epi_returns = []
    for i in range(NB_EPISODES):
        obs = env.reset()
        epi_return = 0
        
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, rew, done, info = env.step(action)
            transition = (obs, action, next_obs, rew, done)
            replay_buffer.append(transition)
            epi_return +=  rew
            agent.learn()
            obs = next_obs
        
        print('Episode | {:5d} | Return | {:5.2f}'.format(i + 1, epi_return))
        epi_returns.append(epi_return)

    # Plot Smoothed Cumulative Reward (Return)
    plt.plot(get_running_mean(epi_returns))

plt.legend(['100', '1000', '10000', '100000', '1000000'])
plt.xlabel('Episode')
plt.ylabel('$G$')
plt.savefig('images/run_lunarlander_with_buffer_q_smooth.png')
plt.show()

# Test Agent (not reproducible)
env = get_lunar_lander()
nb_test_episodes = 0
epi_returns = []
for i in range(nb_test_episodes):
    obs = env.reset()
    env.render()
    time.sleep(0.1)

    done = False
    epi_return = 0
    while not done:
        action = agent.get_action(obs, epsilon=0)
        obs, rew, done, info = env.step(action)
        epi_return += rew
        env.render()
        time.sleep(0.01)

    epi_returns.append(epi_return)

if nb_test_episodes > 0:
    print('Average Episodic Return of ', np.mean(epi_returns))

env.close()
