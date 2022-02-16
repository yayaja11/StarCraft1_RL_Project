"""
여기서 어떤 방식의 스타크래프트 RL을 만들지 정의하는 것 같음
"""

import numpy as np
from gym import spaces
import TorchCraft.starcraft_gym.proto as proto
import TorchCraft.starcraft_gym.gym_utils as utils

import torchcraft as tc
import torchcraft.Constants as tcc
import TorchCraft.starcraft_gym.envs.starcraft_env as sc

DISTANCE_FACTOR = 16
STATE_BUNKER = [(100,39), (112,39),(124,39),(136,39),(148,39),(100,55),(112,55), (124,55),(136,55),(88,63)]

class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0, self_play = False, max_episode_steps = 2000):
        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed, frame_skip, self_play, max_episode_steps)

    def _action_space(self):
        action_low = [0]
        action_high = [9]
        return spaces.Discrete(9)

    def _observation_space(self):
        obs_low  = [1,2,3,4,5,6,7,8,9,0]
        obs_high  = [1,2,3,4,5,6,7,8,9,0]
        print(spaces.Box(np.array(obs_low), np.array(obs_high)))
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands(self, action):
        cmds = []
        if self.state is None or action is None:
            return cmds

        myself_id = None
        myself = None
        enemy_id = None
        enemy = None
        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            return cmds
        # print('buildable?:',self.state.buildable_data)
        # 각각의 유닛이 무엇인지 확인하는 부분
        # for ii in self.state.units[0]:
#        print(issubclass(int, self.state))

        # object_methods = [method_name for method_name in dir(self.state.units[0][0].id)]  # state하위 모든 메쏘드
        # for i in object_methods:
        #     a = [c for c in dir(i)]
        #     print(a)
        # print(object_methods)

        # print('mineral:',self.state.units[0][0].id)
        # if self.state.frame.resources[0].ore >= 100:


        for a in self.state.units[0]:
            if a.type == 7 and a.idle == True:
                print('SCV is idle')

                print(a.idle)
                myself_id = a.id
                myself = a
                break

        for b in self.state.units[1]:
            enemy_id = b.id
            enemy = b

        # if action[0] > 0:
        if myself is None:
            return cmds

        cmds.append([
            # tcc.command_unit_protected, myself_id,
            # tcc.unitcommandtypes.Attack_Move, -1, myself.x -20, myself.y -20
            # ])
            tcc.command_unit_protected, myself_id,
            tcc.unitcommandtypes.Build, -1, STATE_BUNKER[action][0], STATE_BUNKER[action][1], tcc.unittypes.Terran_Supply_Depot])


        # else:
        #     if myself is None:
        #         return cmds

            # print( myself.x, myself.y)
            # print(myself.type)

            # cmds.append([
            #     tcc.command_unit_protected, myself_id,
            #     tcc.unitcommandtypes.Build, -1, myself.x+30-i, myself.y+30-j, tcc.unittypes.Terran_Supply_Depot])  # numpy.float64형식을 받을수없다고해서 int로 바꿈

        return cmds

    def _make_observation(self):
        myself = None
        enemy = None

        obs = np.zeros(self.observation_space.shape)
        print(obs)
        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            return obs
        myunits = self.state.units[0]
        enemy_units = self.state.units[1]

        for a in myunits:
            myself = a

#        for uid, ut in self.state.units[0]:  # 원래 units_myself.items
#            myself = ut
        for b in enemy_units:  # 원래 units_enemy.items():
            enemy = b


        if myself is not None:  # 현재 상태??
            obs[0] = 1
            obs[1] = 1
            obs[2] = 1
            obs[3] = 1
            obs[4] = 1
            obs[5] = 1
            obs[6] = 1
            obs[7] = 1
            obs[8] = 1
            obs[9] = 1
            print(obs)


        return obs

    def _check_done(self):
        return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

    def _compute_reward(self):  #보상 계산
        reward = 0
        if self.obs[5] + 1 > 1.5:
            reward = -1

        if self.obs_pre[6] > self.obs[6]:
            reward = 15
        if self.obs_pre[0] > self.obs[0]:
            reward = -10
        if self._check_done() and not bool(self.state.battle_won):
            reward = -500
        if self._check_done() and bool(self.state.battle_won):
            reward = 1000
            self.episode_wins += 1
        if self.episode_steps == self.max_episode_steps:
            reward = -500

        return reward

    def _get_info(self):
        return {}