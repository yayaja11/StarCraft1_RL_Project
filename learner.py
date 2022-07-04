import time
import pickle
import argparse

from io import BytesIO
from core.algorithm.DQN import Dueling_DQN, DQN
from core.common.replaybuffer import ReplayBuffer, PrioritizedReplayBuffer
from keras.optimizers import Adam
import tensorflow as tf
import zmq
import numpy as np


def update_target_network(self, TAU):
    phi = self.network.get_weights()
    target_phi = self.target_network.get_weights()
    for i in range(len(phi)):
        target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
    self.target_network.set_weights(target_phi)

class learner(object):
    def __init__(self, algorithm, double, dueling, per):
        self.action_n = 67
        self.algorithm = algorithm
        self.double = double
        self.dueling = dueling
        self.per = per
        self.TAU = 0.001

        if self.dueling == True:
            self.network = Dueling_DQN(self.action_n)
            self.target_network = Dueling_DQN(self.action_n)
        else:
            self.network = DQN(self.action_n)
            self.target_network = DQN(self.action_n)

        self.network.build(input_shape=(None, self.state_dim))
        self.target_network.build(input_shape=(None, self.state_dim))

        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        if per == True:
            self.buffer = PrioritizedReplayBuffer(self.BUFFER_SIZE)
        else:
            self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []
        self.miss_action = 0

        update_target_network(1.0)
        self.init_zmq()
        print('press Enter when the actors are ready:')
        input()

        self.publish_model()

    def init_zmq(self):
        context = zmq.Context()

        self.actor_sock = context.socket(zmq.REP)    # model 요청에 따른 응답
        self.actor_sock.bind("tcp://*:10100")

        self.buffer_sock = context.socket(zmq.REQ)
        self.buffer_sock.bind("tcp://*:10101")


    def publish_model(self):
        bio = BytesIO
        self.network.save_weights(bio)
        self.target_network.save_weights(bio)
        self.actor_sock.send(bio.getvalue())

    def train(self):
        idxs = errors = None
        train_cnt = 1
        while True:
            if self.per is True:
                payload = pickle.dumps((idxs, errors))
                if errors is not None:
                    priority = np.mean(errors)
            else:
                payload = b''
            self.buffer_sock.send(payload)
            payload = self.buffer_sock.recv()

            if payload == b'not enough':
                print('not enough')
                time.sleep(1)
            else:
                train_cnt += 1
                states, actions, rewards, next_states,indices, dones = pickle.loads(payload) #  batch새로 받아서 넣는거해야함

                states, actions, rewards, next_states, _, dones = self.buffer.sample_batch((self.BATCH_SIZE))

                if self.double is True:
                    curr_net_qs = self.network(tf.convert_to_tensor(next_states, dtype=tf.float32))  # double에서 행동 뽑는 theta
                    max_a = np.argmax(curr_net_qs.numpy(), aixs=1)

                target_qs = self.target_network(
                    tf.convert_to_tensor(next_states, dtype=tf.float32))  # target_qs는 next_state만 담은 것
                y_i = self.td_target(rewards, target_qs.numpy(), max_a, dones)

                self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32), actions,
                               tf.convert_to_tensor(y_i, dtype=tf.float32))
                self.update_target_network(self.TAU)
                if train_cnt % 40 == 0:
                    self.publish_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', help='rl algorithm', default='DQN')
    parser.add_argument('--double', help = 'applying double', default=False)
    parser.add_argument('--dueling', help='applying dueling', default=False)
    parser.add_argument('--per', help='applying priorities experience replay', default=False)
    args = parser.parse_args()

    L = learner(args.algorithm, args.double, args.dueling, args.per)
    L.train()