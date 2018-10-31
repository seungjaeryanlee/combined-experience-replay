from collections import deque
import random

import numpy as np


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)    

    def append(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class LSTReplayBuffer:
    def __init__(self, long_capacity, short_capacity, error_threshold=0.5):
        self.long_buffer = deque(maxlen=long_capacity)
        self.short_buffer = deque(maxlen=short_capacity)
        self.error_threshold = error_threshold

    def __len__(self):
        return max(len(self.long_buffer), len(self.long_buffer))

    def append(self, state, action, next_state, reward, done, error):
        self.short_buffer.append((state, action, next_state, reward, done, error))

    def cleanup(self):
        while len(self.short_buffer):
            transition = self.short_buffer.pop()
            if transition[-1] > self.error_threshold:
                self.long_buffer.append(transition)

    def sample(self, long_batch_size, short_batch_size):
        long_batch = self.long_sample(long_batch_size)
        short_batch = self.short_sample(short_batch_size)

        return long_batch + short_batch

    def long_sample(self, batch_size):
        return random.sample(self.long_buffer, batch_size)

    def short_sample(self, batch_size):
        return random.sample(self.long_buffer, batch_size)
