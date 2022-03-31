import argparse
import envs.single_battle_env as sc
from core.common.replaybuffer import ReplayBuffer
import bunker_map as bm
import math

import time
import torch
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

DISCOUNT_FACTOR = 0.99
ENABLE_DOUBLE = False
ENABLE_DUELING = False
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0005
TARGET_MODEL_UPDATE_INTERVAL = 100
WINDOW_LENGTH=1
MAX_STEP_CNT = 3000000

class DQN(Model):

    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.h0 = Dense(128, activation='relu')
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(action_n, activation='linear')


    def call(self, x):
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        q = self.q(x)
        return q

class RandomAgent(object):
    def __init__(self, env, state_size, action_size):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000        # 한 게임에 최대 60번의 action을 취하는데 버퍼 사이즈가 20000이면 너무 이전것까지 포함됨. 중복고려해서 조정 state가 다양할수록 더 커야할까 작아야할까? -> 좀 좋은 최근애들이 약 20action한다 가정했을때 500판까지 => 10000
        self.DQN_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.EPSILON = 0.3
        #self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01
        self.env = env

        self.state_dim = state_size
        self.action_n = action_size

        self.dqn = DQN(self.action_n)
        self.target_dqn = DQN(self.action_n)

        self.dqn.build(input_shape=(None, self.state_dim))
        self.target_dqn.build(input_shape=(None, self.state_dim))

        self.dqn.summary()

        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []
        self.miss_action = 0

    def choose_action(self, state):
        temp_mask = self.mask[0].numpy()
        if np.random.random() <= self.EPSILON:
            while 1:
                print('@@ random', self.EPSILON)
                pick = self.env.action_space.sample()
                if temp_mask[pick] == 1:
                    pass
                elif temp_mask[pick] == 0:
                    break

            return pick
        else:
            qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
            np_tensor = qs.numpy()
            qs = tf.where(temp_mask, -100.0, np_tensor)     # masking
            print(qs)
            print(np.argmax(qs.numpy()))
            return np.argmax(qs.numpy())

    def update_target_network(self, TAU):
        phi = self.dqn.get_weights()
        target_phi = self.target_dqn.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_dqn.set_weights(target_phi)

    def dqn_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_n)
            q = self.dqn(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.dqn_opt.apply_gradients(zip(grads, self.dqn.trainable_variables))

    def td_target(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)        # 각각의 next_state들에서 가장 value가 높은 값을 구함
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]):
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * max_q[i]
        return y_k

    def load_weights(self, path):
        self.dqn.load_weights(path + 'random_bunker31413329.h5')

    def time_funct(self, timee):

        # return (1/50000 * (timee ** 2)) - 0.8
        return (7 * ((1/50 * timee) ** 2)) / 10

    def train(self, max_episode_num):
        self.update_target_network(1.0)


        for ep in range(int(max_episode_num)):
            timee, episode_reward, done = 0, 0, False
            steps = -1
            state = self.env._reset()
            next_state = state
            self.mask = torch.zeros(1, 117)

            while not done:
                steps += 1
                action = self.choose_action(state)
                if action == 0 or action == 57:
                    pass
                elif action >= 114:
                    pass
                elif action > 57 and action < 114:
                    self.mask[0][action] = 1
                    self.mask[0][action - 57] = 1

                elif action < 57:
                    self.mask[0][action] = 1
                    self.mask[0][action + 57] = 1

                if action >= 57 and action < 114 and next_state[57+5] == 12:
                    for i in range(57):
                        self.mask[0][57 + i] = 1

                next_state, reward, done, _ = self.env._step(action)
                train_reward = steps * 0.001        # 게임시간이 아니라 STEPS수로 보상
                    # timee = next_state[57 + 6]
                    # train_reward = reward + self.time_funct(timee)       # 시간이 흐를수록 더 높은점수를 주면 나중엔 무슨짓을해도 높은점수를 받아서 막 행동하지 않을까? -> 그만큼 패널티도 증가한다면?
                print('train_reward:', train_reward)
                # if next_state[57+7] == 1:
                #     temp += 1
                #     if temp >= 10:
                #         self.buffer.add_buffer(state, action, train_reward, next_state, done)
                #         temp = 0
                #     else:
                #         pass
                # else:
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                # if self.buffer.buffer_count() > 60:
                if self.buffer.buffer_count() > 1:
                    if self.EPSILON > self.EPSILON_MIN:
                        self.EPSILON *= self.EPSILON_DECAY
                    
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch((self.BATCH_SIZE))
                    target_qs = self.target_dqn(tf.convert_to_tensor(next_states, dtype=tf.float32))    # target_qs는 next_state만 담은 것

                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32), actions, tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.update_target_network(self.TAU)
                state = next_state
                episode_reward += reward        # time reward는 포함되지않음.

            now = time.localtime()
            yy = open('C:\starlog\log_end.txt', 'a')
            yy.write('---------------------------------\n')
            end_data = str(ep) + ': ' + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(
                now.tm_min) + str(now.tm_sec) + 'obs: ' + str(list(map(int, next_state)))
            yy.write(end_data)
            yy.write('\n')
            yy.close()

            print('observation:', list(map(int, next_state)))

            print('reward:', episode_reward)
            print('Episode: ', ep + 1, 'Time: ', timee, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)
            save_name = '../random_bunkerh5/no_reward' + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec) + '.h5'
            self.dqn.save_weights(save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help = 'server ip')
    parser.add_argument('--port', help = 'server port', default ='11111')
    args = parser.parse_args()
    global old_actions
    old_actions = []
    state_size = 57 + 10
    action_size = len(bm.STATE_BUNKER)     # 원래는 환경에서 size들을 반환해주어야 함

    max_episode_num = 10000
    print(args.ip, args.port)
    env = sc.SingleBattleEnv(args.ip, args.port)
    env.seed(123)

    episodes = 0
    agent = RandomAgent(env, state_size, action_size)
    # agent.load_weights('./')
    agent.train(max_episode_num)
