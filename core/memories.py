from TorchCraft.starcraft_gym.core.common.memory import *
import operator
import copy

class SequentialMemory(Memory):
    def __init__(self, limit, enable_per=False, per_alpha=0.6, per_beta=0.4, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        self._next_idx = 0
        self.enable_per = enable_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta

        if self.enable_per:
            if per_alpha < 0 or per_alpha > 1:
                assert False, "per_alpha must be between 0 and 1"
            elif per_beta < 0 or per_beta > 1:
                assert False, "per_beta must be between 0 and 1"

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _sample_proportional(self, time_window, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.observations) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            while idx < time_window:
                mass = random.random() * self._it_sum.sum(0, len(self.observations) -1)
                idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.actions)
            self._it_sum[idx] = priority ** self.per_alpha
            self._it_min[idx] = priority ** self.per_alpha

            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, batch_idxs = None):
        assert self.nb_entries >= self.window_length +2, ' not enouth entires in the memory'

        if batch_idxs is None:
            if self.enable_per:
                batch_idxs = self._sample_proportional(self.window_length, batch_size)

                weights = []
                p_min = self._it_min.min() / self._it_sum.sum()
                max_weight = (p_min * len(self.terminals)) ** (-self.per_beta)

                for idx in batch_idxs:
                    p_sample = self._if_sum[idx] / self._it_sum.sum()
                    weight = (p_sample * len(self.terminals)) ** (-self.per_beta)
                    weights.append(weight / max_weight)
                weights = np.array(weights)
            else:
                batch_idxs = sample_batch_indexes(
                    self.window_length, self.nb_entries - 1, size=batch_size)

        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length +1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                idx = sample_batch_indexes(self.window_length +1, self.no_entries, size=1)[0]
                terminal0 = self.terminal[idx - 2]
            assert self.window_length +1 <= idx < self.nb_entries

            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    break
                state0.insert(0, self.observations[current_idx])

            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx -1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx -1]
            state1 = [copy.deepcopy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0= state0, action = action, reward = reward,
                                          state1= state1, terminal1=terminal1))

        assert len(experiences) == batch_size
        if self.enable_per:
            return experiences, weights, batch_idxs
        else:
            return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)

        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            if self.enable_per:
                idx = self._next_idx
                self._it_sum[idx] = self._max_priority ** self.per_alpha
                self._it_min[idx] = self._max_priority ** self.per_alpha

                if self._next_idx >= len(self.observations):
                    self.observations.append(observation)
                    self.actions.append(action)
                    self.rewards.append(reward)
                    self.terminals.append(terminal)
                else:
                    self.observations.data[self._next_idx] = observation
                    self.actions.data[self._next_idx] = action
                    self.rewards.data[self._next_idx] = reward
                    self.terminals.data[self._next_idx] = terminal
                self._next_idx = (self._next_idx + 1) % self.limit
            else :
                self.observations.append(observation)
                self.actions.append(action)
                self.rewards.append(reward)
                self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):

        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).reduce(start, end)