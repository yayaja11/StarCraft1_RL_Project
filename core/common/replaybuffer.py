import numpy as np
from collections import deque
import random
import time
import zmq
import pickle

import argparse
# buffer이름이 actor의 buffer와 겹쳐서 _붙임


def async_recv(sock):
    try:
        return sock.recv(zmq.DONTWAIT)
    except zmq.Again:
        pass



class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer_ = deque()
        self.count = 0

    def add_buffer(self,state, action,reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.count < self.buffer_size:
            self.buffer_.append(transition)
            self.count += 1
        else:
            self.buffer_.popleft()
            self.buffer_.append(transition)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer_, self.count)
        else:
            batch = random.sample(self.buffer_, batch_size)
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer_ = deque()
        self.count = 0

class PrioritizedReplayBuffer(object): 
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer_size = buffer_size
        self.buffer_ = deque()
        self.pos = 0
        self.priorities = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.alpha = alpha
        self.beta = 0.4

    def update_beta(self, idx):
        v = 0.4 + idx * (1.0 - 0.4) / 100000
        self.beta = min(1.0, v)
        return self.beta

    def add_buffer(self,state, action,reward, next_state, done):    # prio에서 replay 추가 저장
        max_prio = self.priorities.max() if self.buffer_ else 1.0
        transition = (state, action, reward, next_state, done)

        if self.count < self.buffer_size:
            self.buffer_.append(transition)
            self.count += 1

        else:
            self.buffer_.popleft()
            self.buffer_.append(transition)
        self.priorities[self.count] = max_prio                      # 각각의 buffer는 priority를 가지고 있음음

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            prios = self.priorities[:self.count]
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(self.count, self.count, p=probs)
        else:
            prios = self.priorities
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(self.count, batch_size, p=probs)
        # weights = (self.count * probs[indices]) ** (-self.beta)
        # weights /= weights.max()

        # indices = np.random.choice(self.count, batch_size, p=probs)

        states = np.asarray([i[0] for i in indices])
        actions = np.asarray([i[1] for i in indices])
        rewards = np.asarray([i[2] for i in indices])
        next_states = np.asarray([i[3] for i in indices])
        dones = np.asarray([i[4] for i in indices])

        # return states, actions, rewards, next_states, dones, np.array(weights, dtype=np.float32)
        return states, actions, rewards, next_states, indices, dones


    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.priorities = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.count = 0


    def take_all(self):
        return self.buffer, self.priorities


if __name__ == '__main__':
    BUFFER_SIZE = 100000
    MIN_SIZE = 10000
    BATCH_SIZE = 64
    parser = argparse.ArgumentParser()
    parser.add_argument('--per', help='applying priorities experience replay', default=False)
    args = parser.parse_args()

    buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

    context = zmq.Context()
    learner_sock = context.socket(zmq.REP)
    learner_sock.connect("tcp://*:10001")  # get model from learner

    actor_sock = context.socket(zmq.PULL)
    actor_sock.connect("tcp://*:10000")  # send exp to buffer

    while True:
        payload_act = async_recv(actor_sock)
        if payload_act is not None:
            st = time.time()
            if args.per == True:
                actor_id, batch, prios = pickle.loads(payload_act)
            else:
                actor_id, batch = pickle.loads(payload_act)

        payload_lrn = async_recv(learner_sock)
        if payload_lrn is not None:
            idxs, errors = pickle.loads(payload_lrn)
            if idxs is not None:
                buffer.update_priorities(idxs, prios)

        if buffer.buffer_count() < MIN_SIZE:
            payload = b'not enough'
        else:
            states, actions, rewards, next_states,indices, dones = buffer.sample_batch(BATCH_SIZE)
            payload_send = pickle.dumps((states, actions, rewards, next_states,indices, dones))

        learner_sock.send(payload_send)

