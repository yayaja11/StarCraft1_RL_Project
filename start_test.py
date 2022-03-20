import argparse
# import TorchCraft.starcraft_gym.envs.single_battle_env as sc
import starcraft_gym.envs.single_battle_env as sc
# from TorchCraft.starcraft_gym.core.common.replaybuffer import ReplayBuffer
from starcraft_gym.core.common.replaybuffer import ReplayBuffer
import math

import time
import torch
import datetime
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import keras.backend as Kb
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
import keras.optimizers as optimizers
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

from tqdm import tqdm
from gym import spaces

DISCOUNT_FACTOR = 0.99
ENABLE_DOUBLE = False
ENABLE_DUELING = False
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0005
TARGET_MODEL_UPDATE_INTERVAL = 100
WINDOW_LENGTH=1
MAX_STEP_CNT = 3000000


STATE_BUNKER = [(0,0,3), (112,39,0),(124,39,0),(136,39,0),(148,39,0)
                                                               ,(156, 47,0)
                ,(100,55,0),(112,55,0),(124,55,0),(136,55,0),         (156, 55,0)
        ,(88,63,0),(100,63,0),(112,63,0),(124,63,0),(136,63,0),         (156, 63,0),(168,63,0)
,(80,71,0)
,(80,79,0),       (100, 79,0),(112,79,0),(124,79,0),(136,79,0),         (156, 79,0),(168,78,0)
,(80,87,0),       (100, 87,0),(112,87,0),(124,87,0),(136,87,0),         (156, 87,0),(168,87,0)
,(80,95,0),       (100, 95,0),(112,95,0),(124,95,0),(136,95,0),         (156, 95,0),(168,95,0)
,(80,103,0),      (100, 103,0),(112,103,0),(124,103,0),(136,103,0),     (156, 103,0),(168,103,0)
,(80,111,0),                                                            (156, 111,0),(168,111,0)
    ,(84,119,0),(96,119,0),(108,119,0),(120,119,0),(132,119,0),(144,119,0),(156,119,0)
                ,
                (0,0,3), (112,39,1),(124,39,1),(136,39,1),(148,39,1)
                                                               ,(156, 47,1)
                ,(100,55,1),(112,55,1),(124,55,1),(136,55,1),         (156, 55,1)
        ,(88,63,1),(100,63,1),(112,63,1),(124,63,1),(136,63,1),         (156, 63,1),(168,63,1)
,(80,71,1)
,(80,79,1),       (100, 79,1),(112,79,1),(124,79,1),(136,79,1),         (156, 79,1),(168,78,1)
,(80,87,1),       (100, 87,1),(112,87,1),(124,87,1),(136,87,1),         (156, 87,1),(168,87,1)
,(80,95,1),       (100, 95,1),(112,95,1),(124,95,1),(136,95,1),         (156, 95,1),(168,95,1)
,(80,103,1),      (100, 103,1),(112,103,1),(124,103,1),(136,103,1),     (156, 103,1),(168,103,1)
,(80,111,1),                                                            (156, 111,1),(168,111,1)
    ,(84,119,1),(96,119,1),(108,119,1),(120,119,1),(132,119,1),(144,119,1),(156,119,1)
                ,(0,0,3),(0,0,3),(0,0,3)]

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
                print(temp_mask)
                pick = self.env.action_space.sample()
                if temp_mask[pick] == 1:
                    pass
                elif temp_mask[pick] == 0:
                    break

            return pick
        else:
            qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
            np_tensor = qs.numpy()
            # print('@@\n',np_tensor[0])
            # print('@@\n',temp_mask)
            # print('@@\n',len(np_tensor[0]))
            # print('@@\n',len(temp_mask))

            qs = tf.where(temp_mask, -100.0, np_tensor)
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
                # temp = 0        # 중복 replay buffer에 걸러서 넣기 위함

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
                if self.buffer.buffer_count() > 6000:
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
            save_name = './random_bunkerh5/no_reward' + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec) + '.h5'
            self.dqn.save_weights(save_name)

    # def act(self):
    #         # while 1:  # 임시로 유니크 타워 2개지으면 해당 action 하지않도록하고 rl로 학습할 땐 3개지으려할때 패널티를줘서 안짓게 뭐가 좋을지 고민 # 유니크는 10라운드이후부터
    #         # 시작되는 특수한 상황이라 네트워크를 두개써야할 수 있음 일단 영웅벙커까지만.
    #         # if self.post_action == False:
    #         #     action_t = 115
    #
    #         action_t = self.forward(obs)  # action 확률기반 선택을 넣을 곳
    #         print('action', action_t, old_actions)
    #         reward = 0.1
    #         self.backward(reward, terminal=True)
    #         # action_t = myaction.pop()
    #         while action_t in old_actions:      # 중복장소에 짓는것 방지. 여기서 처리하는 이유는 환경에서하면 scv가 일을 안하는 이유를 알아야하는데 버그/중복/대기상태 를 특정할 수 가 없음
    #
    #             action_t = self.forward(obs)
    #             self.miss_action -= 1
    #             reward = -100
    #             self.backward(reward, terminal = True)
    #
    #         # self.episode_reward += reward  # 이거 필요
    #
    #         old_actions.append(action_t)
    #         if action_t == 114:
    #             pass
    #         elif action_t >= 57 and action_t < 114:
    #             print(action_t - 57)
    #             old_actions.append(action_t - 57)
    #         elif action_t < 57:
    #             print(action_t + 57)
    #             old_actions.append(action_t + 57)
    #
    #         return action_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help = 'server ip')
    parser.add_argument('--port', help = 'server port', default ='11111')
    args = parser.parse_args()
    global old_actions
    old_actions = []
    state_size = 57 + 10
    action_size = len(STATE_BUNKER)     # 원래는 환경에서 size들을 반환해주어야 함

    max_episode_num = 10000
    print(args.ip, args.port)
    env = sc.SingleBattleEnv(args.ip, args.port)
    env.seed(123)

    episodes = 0
    agent = RandomAgent(env, state_size, action_size)
    # agent.load_weights('./')
    agent.train(max_episode_num)

    #
    #
    # while episodes < 150:
    #     old_actions = []
    #     obs = env._reset()
    #     done = False
    #
    #     # for _ in tqdm(range(1000)):
    #     while not done:
    #         action = agent.act()
    #         obs, reward, done, info = env._step(action)

        # episodes += 1
    #     if done:
    #         agent.forward(obs)
    #         agent.backward(0., terminal=False)
    #
    # env.close()