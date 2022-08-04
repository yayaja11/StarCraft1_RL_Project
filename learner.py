import time
import pickle
import argparse

from io import BytesIO
# from core.algorithm.DQN import Dueling_DQN, DQN
# from core.common.replaybuffer import ReplayBuffer, PrioritizedReplayBuffer
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from collections import deque
import random
import DQN as dqn
import bunker_map as bm

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = deque(maxlen=self.capacity)
        self.count = 0

    def buffer_count(self):
        return self.count

    def add_buffer(self,state, action,reward, next_state, done):

        transition = (state, action, reward, next_state, done)

        if self.count < self.capacity:
            self.data.append(transition)
            self.count += 1
        else:
            self.data.popleft()
            self.data.append(transition)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.data, self.count)
        else:
            batch = random.sample(self.data, batch_size)
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones

    def clear_buffer(self):
        self.data = deque(maxlen=self.capacity)
        self.count = 0


def dqn_learn(network, action_size, states, actions, td_targets):
    with tf.GradientTape() as tape:
        one_hot_actions = tf.one_hot(actions, action_size)
        q = network(states, training=True)
        q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
        loss = tf.reduce_mean(tf.square(q_values-td_targets))

    grads = tape.gradient(loss, network.trainable_variables)
    dqn_opt = Adam(0.001)   # learning rate
    dqn_opt.apply_gradients(zip(grads, network.trainable_variables))

def td_target(action_size, rewards, target_qs, max_a, dones, double):
    if double == True:
        one_hot_max_a = tf.one_hot(max_a, action_size)
        max_q = np.max(one_hot_max_a * target_qs, axis=1, keepdims=True)
    else:
        max_q = np.max(target_qs, axis=1, keepdims=True)        # 각각의 next_state들에서 가장 value가 높은 값을 구함

    y_k = np.zeros(max_q.shape)
    for i in range(max_q.shape[0]):
        if dones[i]:
            y_k[i] = rewards[i]
        else:
            y_k[i] = rewards[i] + 0.95 * max_q[i] # gamma
    return y_k

def update_target_network(TAU, network, target_network):
    phi = network.get_weights()
    target_phi = target_network.get_weights()
    for i in range(len(phi)):
        target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
    target_network.set_weights(target_phi)

def learner(q, param_q, double, dueling, batch_size):
    action_size = len(bm.STATE_BUNKER)
    memory = Memory(40000)
    state_dim = 57 + 10
    if dueling == True:
        network = dqn.Dueling_DQN(action_size)
        target_network = dqn.Dueling_DQN(action_size)

    else:
        network = dqn.DQN(action_size)
        target_network = dqn.DQN(action_size)

    network.build(input_shape=(None, state_dim))
    target_network.build(input_shape=(None, state_dim))

    while True:
        while not q.empty():
            temp = q.get()
            state = temp[0]
            action = temp[1]
            reward = temp[2]
            next_state = temp[3]
            done = temp[4]

            memory.add_buffer(state, action, reward, next_state, done)
        if memory.count > 10000: # memory buffer count랑 count랑 다른가? 아무튼 여기 진입못하는듯
            states, actions, rewards, next_states, dones = memory.sample_batch((batch_size))

            if double is True:
                curr_net_qs = network(tf.convert_to_tensor(next_states, dtype=tf.float32))  # double에서 행동 뽑는 theta
                max_a = np.argmax(curr_net_qs.numpy(), axis=1)

            target_qs = target_network(tf.convert_to_tensor(next_states, dtype=tf.float32))  # target_qs는 next_state만 담은 것
            y_i = td_target(action_size, rewards, target_qs.numpy(), max_a, dones, double)

            dqn_learn(network, action_size, tf.convert_to_tensor(states, dtype=tf.float32), actions,
                           tf.convert_to_tensor(y_i, dtype=tf.float32))
            update_target_network(0.001, network, target_network) # TAU

            # 만약 뺏을 때 actor가 마침 param_q에서 빼고 다시 넣는중일 수 있음 그때를 방지하기 위함
            if not param_q.empty():
                try:
                    garbege = param_q.get()
                except:
                    pass

            param_q.put([network.get_weights(), target_network.get_weights()]) # main, target


