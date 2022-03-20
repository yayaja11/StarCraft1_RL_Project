import gym
import numpy as np
import tensorflow as tf
from TorchCraft.start_test import RandomAgent
import TorchCraft.starcraft_gym.envs.single_battle_env as sc
import argparse

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
                (100,39,1), (112,39,1),(124,39,1),(136,39,1),(148,39,1)
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help = 'server ip')
    parser.add_argument('--port', help = 'server port', default ='11111')
    args = parser.parse_args()

    state_size = 57 + 10
    action_size = len(STATE_BUNKER)     # 원래는 환경에서 size들을 반환해주어야 함

    max_episode_num = 200
    print(args.ip, args.port)
    env = sc.SingleBattleEnv(args.ip, args.port)

    print(env.observation_space.shape[0])  # 4
    # get action dimension
    print(env.action_space, env.observation_space)

    state_size = 57 + 10
    action_size = len(STATE_BUNKER)


    env = sc.SingleBattleEnv(args.ip, args.port)
    env.seed(123)
    agent = RandomAgent(env,state_size, action_size)

    agent.load_weights('../')

    time = 0

    while True:
        state = env._reset()
        qs = agent.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
        action = np.argmax(qs.numpy())

        state, reward, done, _ = env._step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        if done:
            break

if __name__=="__main__":
    main()