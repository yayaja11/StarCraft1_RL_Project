import argparse
import envs.make_command_env as sc
from core.common.replaybuffer import ReplayBuffer
import bunker_map as bm
import torch
import DQN as dqn

import time
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from keras.optimizers import Adam
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 0.0005
TARGET_MODEL_UPDATE_INTERVAL = 100
WINDOW_LENGTH=1
MAX_STEP_CNT = 3000000


class DQNAgent(object):
    def __init__(self, env, state_size, action_size, algorithm, double, dueling):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000        # 한 게임에 최대 60번의 action을 취하는데 버퍼 사이즈가 20000이면 너무 이전것까지 포함됨. 중복고려해서 조정 state가 다양할수록 더 커야할까 작아야할까? -> 좀 좋은 최근애들이 약 20action한다 가정했을때 500판까지 => 10000
        self.DQN_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.995
        self.EPSILON_MIN = 0.01
        self.env = env
        self.algorithm = algorithm
        self.double = double
        self.dueling = dueling

        self.state_dim = state_size
        self.action_n = action_size

        if self.dueling == True:
            self.network = dqn.Dueling_DQN(self.action_n)
            self.target_network = dqn.Dueling_DQN(self.action_n)

        else:
            self.network = dqn.DQN(self.action_n)
            self.target_network = dqn.DQN(self.action_n)

        self.network.build(input_shape=(None, self.state_dim))
        self.target_network.build(input_shape=(None, self.state_dim))

        self.network.summary()

        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []
        self.miss_action = 0

    def update_target_network(self, TAU):
        phi = self.network.get_weights()
        target_phi = self.target_network.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_network.set_weights(target_phi)


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
            qs = self.network(tf.convert_to_tensor([state], dtype=tf.float32))
            np_tensor = qs.numpy()
            qs = tf.where(temp_mask, -500.0, np_tensor)     # masking
            # print(qs)
            # print(np.argmax(qs.numpy()))
            return np.argmax(qs.numpy())

    # def dqn_learn(self, states, actions, td_targets):
    #     with tf.GradientTape() as tape:
    #         one_hot_actions = tf.one_hot(actions, self.action_n)
    #         q = self.network(states, training=True)
    #         q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
    #         loss = tf.reduce_mean(tf.square(q_values-td_targets))
    #
    #     grads = tape.gradient(loss, self.network.trainable_variables)
    #     self.dqn_opt.apply_gradients(zip(grads, self.network.trainable_variables))

    # def td_target(self, rewards, target_qs, max_a, dones):
    #     if self.double == True:
    #         one_hot_max_a = tf.one_hot(max_a, self.action_n)
    #         max_q = np.max(one_hot_max_a * target_qs, axis=1, keepdims=True)
    #     else:
    #         max_q = np.max(target_qs, axis=1, keepdims=True)        # 각각의 next_state들에서 가장 value가 높은 값을 구함
    #
    #     y_k = np.zeros(max_q.shape)
    #     for i in range(max_q.shape[0]):
    #         if dones[i]:
    #             y_k[i] = rewards[i]
    #         else:
    #             y_k[i] = rewards[i] + self.GAMMA * max_q[i]
    #     return y_k

    def load_weights(self, path):
        self.network.load_weights(path + 'random_bunker31413329.h5')

    def time_funct(self, timee):

        # return (1/50000 * (timee ** 2)) - 0.8
        return (7 * ((1/50 * timee) ** 2)) / 10

    def play_step(self, max_episode_num, q, param_q, x): # x = actor 번호
        self.update_target_network(1.0)
        total_step = 0
        for ep in range(int(max_episode_num)):
            timee, episode_reward, done = 0, 0, False
            max_a = 0
            steps = -1
            state = self.env._reset()
            next_state = state
            self.mask = torch.zeros(1, 117)         # torch 쓴 이유: tensorflow zeros는 생성한 리스트 변경불가

            while not done:
                steps += 1
                total_step += 1
                action = self.choose_action(state)
                if action == 0 or action == 57:     # 이 장소에 벙커를 지으면 버그 발생
                    pass
                elif action >= 114:                 # 업그레이드는 masking X
                    pass
                elif action > 57 and action < 114:  # 행동을 정했을 경우 해당 장소를 masking 하여 재선택하지 않도록
                    self.mask[0][action] = 1        # 가장 이상적인 것은 건설이 완료됐을 때 하는 것 이지만 건설이 완료됬음을 train까지 반환하기는 과하다 판단
                    self.mask[0][action - 57] = 1   # 이렇게해도 반드시 행동을 완수하도록 만들었기 때문에 문제 없음

                elif action < 57:
                    self.mask[0][action] = 1
                    self.mask[0][action + 57] = 1

                if action >= 57 and action < 114 and next_state[57+5] == 12:    # 영웅벙커는 최대 12회까지 지을 수 있습니다.
                    for i in range(57):
                        self.mask[0][57 + i] = 1

                next_state, reward, done, _ = self.env._step(action)
                # train_reward = next_state[57+6] * 0.001 # 기존 step에서 시간으로
                # timee = next_state[57 + 6]
                # train_reward = reward + self.time_funct(timee)       # 시간이 흐를수록 더 높은점수를 주면 나중엔 무슨짓을해도 높은점수를 받아서 막 행동하지 않을까? -> 그만큼 패널티도 증가한다면?
                q.put([state, action, reward, next_state, done])

                #
                # self.buffer.add_buffer(state, action, train_reward, next_state, done)       # 6000개 넘으면 밀어내기
                # print('now exp',state, action, train_reward, next_state, done)

                # if self.buffer.buffer_count() > 6000:
                if total_step > 10:

                    if self.EPSILON > self.EPSILON_MIN:
                        self.EPSILON *= self.EPSILON_DECAY
                    # states, actions, rewards, next_states, dones = self.buffer.sample_batch((self.BATCH_SIZE))

                    # if self.double is True:
                    #     curr_net_qs = self.network(tf.convert_to_tensor(next_states, dtype=tf.float32)) # double에서 행동 뽑는 theta
                    #     max_a = np.argmax(curr_net_qs.numpy(), aixs=1)
                    #
                    # target_qs = self.target_network(tf.convert_to_tensor(next_states, dtype=tf.float32))    # target_qs는 next_state만 담은 것
                    # y_i = self.td_target(rewards, target_qs.numpy(), max_a, dones)
                    #
                    # self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32), actions, tf.convert_to_tensor(y_i, dtype=tf.float32))
                    if not param_q.empty():
                        try:
                            weights = param_q.get()
                            param_q.put(weights)  # 다른 actor들도 동일한 최신 param을 사용할 수 있게
                            self.network.set_weights(weights[0])
                            self.target_network.set_weights(weights[1])
                            self.update_target_network(self.TAU)
                        except:
                            pass
                #
                state = next_state
                episode_reward += reward        # time reward는 포함되지않음.


            now = time.localtime()
            yy = open('C:\starlog\llog_8multi_ver6.txt', 'a')
            yy.write('---------------------------------\n')
            end_data = x + str(ep) + ': ' + str(now.tm_mon)+ str(now.tm_mday) + str(now.tm_hour) + str(
                now.tm_min) + str(now.tm_sec) + 'obs: ' + str(list(map(int, next_state)))
            yy.write(end_data)
            yy.write('\n')
            yy.close()

            # print('observation:', list(map(int, next_state)))

            # print('reward:', episode_reward)
            # print('Episode: ', ep + 1, 'Time: ', timee, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)
            save_name = '../random_bunkerh5/no_reward' + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec) + x +'.h5'
            self.network.save_weights(save_name)

def actor_func(ip, port, algorithm, double, dueling, q, param_q, x ):
    state_size = 57 + 10
    action_size = len(bm.STATE_BUNKER)
    max_episode_num = 10000
    env = sc.MakeCommandEnv(ip, port)
    env.seed(123)
    agent = DQNAgent(env, state_size, action_size, algorithm, double, dueling)
    agent.play_step(max_episode_num, q, param_q, x)    # RL 알고리즘 시작
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ip', help='server ip', default='127.0.0.1')
#     parser.add_argument('--port', help='server port', default ='11111')
#     parser.add_argument('--algorithm', help='rl algorithm', default='DQN')
#     parser.add_argument('--double', help='applying double', default=True)
#     parser.add_argument('--dueling', help='applying dueling', default=True)
#     args = parser.parse_args()
#     global old_actions
#     old_actions = []
#     state_size = 57 + 10
#     action_size = len(bm.STATE_BUNKER)     # 원래는 환경에서 size들을 반환해주어야 함
#
#     max_episode_num = 10000
#     print(args.ip, args.port)
#     env = sc.MakeCommandEnv(args.ip, args.port)
#     env.seed(123)
#
#     episodes = 0
#     agent = DQNAgent(env, state_size, action_size, args.algorithm, args.double, args.dueling)
#     # agent.load_weights('./')
#     agent.train(max_episode_num)    # RL 알고리즘 시작
