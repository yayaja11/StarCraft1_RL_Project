"""
여기서 어떤 방식의 스타크래프트 RL을 만들지 정의하는 것 같음
"""

import numpy as np
from gym import spaces
import TorchCraft.starcraft_gym.proto as proto
import TorchCraft.starcraft_gym.gym_utils as utils

import torchcraft.Constants as tcc
import TorchCraft.starcraft_gym.envs.starcraft_env as sc

DISTANCE_FACTOR = 16

class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0, self_play = False, max_episode_steps = 2000):
        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed, frame_skip, self_play, max_episode_steps)

    def _action_space(self):
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        obs_low  = [  0.0,   0.0, 0.0, 0.0, -1.0,  0.0,  0.0,   0.0,  0.0, 0.0]
        obs_high = [100.0, 100.0, 1.0, 1.0,  1.0, 50.0, 100.0, 100.0, 1.0, 1.0]
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
        # 각각의 유닛이 무엇인지 확인하는 부분
        # for ii in self.state.units[0]:
        #     print(ii.type)
#        print(issubclass(int, self.state))

        object_methods = [method_name for method_name in dir(self.state.frame.resources[0].ore)]  # state하위 모든 메쏘드
        # for i in object_methods:
        #     a = [c for c in dir(i)]
        #     print(a)
        print(object_methods)

        print('mineral:',self.state.frame.resources[0].ore)

        for a in self.state.units[0]:
            if a.type == 7 and a.idle == True:
                print('SCV is idle')

                print(a.idle)
                myself_id = a.id
                myself = a

        for b in self.state.units[1]:
            enemy_id = b.id
            enemy = b

        if action[0] > 0:
            if myself is None or enemy is None:
                return cmds
            cmds.append([
                tcc.command_unit_protected, myself_id,
                tcc.unitcommandtypes.Attack_Unit, enemy_id])
        else:
            if myself is None or enemy is None:
                return cmds
            degree = action[1] * 180  # 180의 의미??
            distance = (action[2] + 1) * DISTANCE_FACTOR # DISTANCE_FACTOR의 의미?
            x2, y2 = utils.get_position(degree, distance, myself.x, -myself.y)  # myself.x -> myself.position[1]
            cmds.append([
                tcc.command_unit_protected, myself_id,
                tcc.unitcommandtypes.Move, -1, int(x2), int(y2)])  # numpy.float64형식을 받을수없다고해서 int로 바꿈

        return cmds

    def _make_observation(self):
        myself = None
        enemy = None

        obs = np.zeros(self.observation_space.shape)

        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            obs[9] = 1.0
            return obs
        myunits = self.state.units[0]
        enemy_units = self.state.units[1]

        for a in myunits:
            myself = a

#        for uid, ut in self.state.units[0]:  # 원래 units_myself.items
#            myself = ut
        for b in enemy_units:  # 원래 units_enemy.items():
            enemy = b


        if myself is not None and enemy is not None:  # 현재 상태??
            obs[0] = myself.health
            obs[1] = myself.groundCD
            obs[2] = myself.groundRange / DISTANCE_FACTOR -1
            obs[3] = 0.0
            obs[4] = utils.get_degree(myself.x, -myself.y, enemy.x, -enemy.y) / 180
            obs[5] = utils.get_distance(myself.x, -myself.y, enemy.x, -enemy.y) / DISTANCE_FACTOR - 1
            obs[6] = enemy.health
            obs[7] = enemy.groundCD
            obs[8] = enemy.groundRange / DISTANCE_FACTOR -1
            obs[9] = 1.0
        else:
            obs[9] = 1.0

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