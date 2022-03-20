from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import numpy as np

Experience = namedtuple('Experience', 'state0, action, reward, strate1, terminal1')

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def __len__(self):
        return self.length()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length():
            raise KeyError()
        return self.data[idx]

    def append(self, v):
        self.data.append(v)

    def length(self):
        return len(self.data)

class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs = None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training = True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length -1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx -1] if current_idx -1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config


def zeroed_observation(observation):

    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.

def sample_batch_indexes(low, high, size):
    if high - low >= size:

        try:
            r = range(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:

        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs