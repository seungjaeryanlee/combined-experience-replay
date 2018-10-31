import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproximateAgent:
    """
    A template class for deep reinforcement learning agents that uses neural
    networks to approximate Q-values with epsilon-greedy exploration.
    """
    def __init__(self, q_network, state_space, action_space, epsilon):
        self.q_network = q_network
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon

    def get_action(self, state, epsilon=None):
        """
        Get action chosen by the agent given a state.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.random() < epsilon: # Random selection
            return np.random.choice(self.action_space.n)
        else: # Greedy selection
            state = torch.FloatTensor(state)
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
            return action

    def get_state_value(self, state):
        """
        Get state value V(s) of given state s.
        """
        return self.q_network(torch.FloatTensor(state)).max().item()

class NNOnlineQAgent(ApproximateAgent):
    def __init__(self, q_network, optimizer, state_space, action_space, epsilon=0.1, gamma=0.999):
        """
        Agent that learns action values (Q) through Q-learning method. Assumes
        Box state space and Discrete action space.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma

        self.q_network = q_network
        self.optimizer = optimizer

    def learn(self, state, action, next_state, reward, done):
        """
        Train the agent with a given transition.
        """
        predicted_q_value = self.q_network(torch.FloatTensor(state))[action]
        with torch.no_grad():
            target_q_value = torch.FloatTensor([reward])
            if not done:
                target_q_value += self.gamma * self.q_network(torch.FloatTensor(next_state)).max().item()
        loss = F.smooth_l1_loss(predicted_q_value, target_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class NNBufferQAgent(ApproximateAgent):
    def __init__(self, q_network, optimizer, replay_buffer, state_space, action_space, epsilon=0.1, gamma=0.999, batch_size=16, min_buffer=0):
        """
        Agent that learns action values (Q) through Q-learning method. Assumes
        Box state space and Discrete action space.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer

        self.q_network = q_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer

    def learn(self):
        """
        Train the agent with a given transition.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        if len(self.replay_buffer) < self.min_buffer:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        done_batch = torch.FloatTensor(done_batch)

        predicted_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # print('T1: ', self.q_network(next_state_batch).max(1)[0])
            # print('T2: ', torch.FloatTensor(1) - done_batch)
            target_q_values = reward_batch + self.gamma * self.q_network(next_state_batch).max(1)[0] * (torch.FloatTensor(1) - done_batch)

        # print('P: ', predicted_q_values)
        # print('T: ', target_q_values)
        loss = F.smooth_l1_loss(predicted_q_values, target_q_values)
        # print('L: ', loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
