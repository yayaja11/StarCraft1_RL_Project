"""
여기서 어떤 방식의 스타크래프트 RL을 만들지 정의하는 것 같음
"""

import math
import numpy as np
from gym import spaces
# import TorchCraft.starcraft_gym.proto as proto
# import TorchCraft.starcraft_gym.gym_utils as utils
import starcraft_gym.proto as proto
import starcraft_gym.gym_utils as utils

import torchcraft as tc
# import torchcraft.Constants as tcc
import torchcraft.Constants as tcc
import starcraft_gym.envs.starcraft_env as sc
# import torchcraft.Constants as tcc
# import TorchCraft.starcraft_gym.envs.starcraft_env as sc


DISTANCE_FACTOR = 16
# (x축, y축, z) z: 0: 일반벙커, 1: 영웅벙커, 2: 유니크 벙커 3: upgrade
STATE_BUNKER = [(0,0,3), (112,39,0),(124,39,0),(136,39,0),(148,39,0)
                                                                     ,(156, 47,0)
                ,(100,55,0),(112,55,0),(124,55,0),(136,55,0),           (156, 55,0)
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



class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0, self_play = False, max_episode_steps = 2000):
        # speed 60= 1000프레임을 60초동안. 1=1000프레임을 1초동안 아닌듯 1~6가는데 빨라짐
        self.speed = speed
        self.curr_upgrade = 0
        self.miss_action = 0
        self.hydra_switch = 0
        self.post_action = []
        self.scv_working = 0
        self.number_of_normal_bunker = 0
        self.number_of_hero_bunker = 0

        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed, frame_skip, self_play, max_episode_steps)

    def _action_space(self):

        self.number_of_state = len(STATE_BUNKER)  # + 영웅 + 유니크 + 업그레이드

        return spaces.Discrete(self.number_of_state)

    def _observation_space(self):
        obs_low  = [1,2,3,4,5,6,7,8,9,10] + [0] * 57
        obs_high  = [1,2,3,4,5,6,7,8,9,10] + [0] * 57
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands_click(self, action, x, y):
        cmds = []
        for a in self.state.units[0]:

            if a.type == 7:
                scv_id = a.id
                scv = a

        print('action:', action)
        if action == ['finish']:
            cmds.append([
                tcc.command_unit_protected, scv_id,
                tcc.unitcommandtypes.Right_Click_Position , -1, x+2, y])

        return cmds

    def _make_commands(self, action, action_num):
        cmds = []
        scv = None
        engineeringbay = None
        hydra = None
        larva = None

        for a in self.state.units[0]:

            if a.type == 7:
                scv_id = a.id
                scv = a

            if a.type == 122:
                engineeringbay_id = a.id
                engineeringbay = a

            if a.type == 35:
                larva_id = a.id
                larva = a

            if a.type == 38:
                hydra_id = a.id
                hydra = a

            if a.type == 109:
                supply_id = a.id
                supply = a

        if action == ['hydra']:
            try:
                print('action:', action)
                print(self.hydra_switch, action)
                print('morph')
                cmds.append([
                    tcc.command_unit, hydra_id,
                    tcc.unitcommandtypes.Morph, -1, -1, -1, tcc.unittypes.Zerg_Lurker])
            except:
                self.hydra_switch = 1

        if action == ['halt']:
            print('action:', action)
            print(self.hydra_switch, action)
            print('halt')
            cmds.append([
                tcc.command_unit, supply_id,
                tcc.unitcommandtypes.Cancel_Construction, -1, -1, -1, -1])

        elif action == ['upgrade']:
            cmds.append([
                tcc.command_unit, engineeringbay_id,
                tcc.unitcommandtypes.Upgrade, -1, -1,-1, tcc.upgradetypes.Terran_Infantry_Weapons])

        elif action == ['upgrade_cancel']:
            cmds.append([
                tcc.command_unit_protected, engineeringbay_id,
                tcc.unitcommandtypes.Cancel_Upgrade, -1 , -1, -1 , tcc.upgradetypes.Terran_Infantry_Weapons])

        elif action == ['bunker']:
            cmds.append([
                tcc.command_unit_protected, scv_id,
                tcc.unitcommandtypes.Build, -1, STATE_BUNKER[action_num][0], STATE_BUNKER[action_num][1],
                tcc.unittypes.Terran_Supply_Depot])

        return cmds

    def _make_observation(self):
        myself = None

        obs = np.zeros(self.observation_space.shape)
        lifes = 0
        lucks = 0
        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            return obs

        for a in self.state.units[0]:
            if a.type == 13:
                lifes += 1

            if a.type == 218:
                lucks += 1

        print('frame_from_bwapi:', self.state.frame_from_bwapi)
        lucks = 0
        for i in self.state.frame.units[0]:
            if i.type == 37:
                lucks += 1

        for i in range(57):
            obs[i] = self.bunker_build_state[i]
        obs[57+1] = lifes  # 라이프
        obs[57+2] = lucks  # 럭
        obs[57+3] = self.curr_upgrade
        obs[57+4] = self.number_of_normal_bunker
        obs[57+5] = self.number_of_hero_bunker
        obs[57+6] = self.state.frame_from_bwapi / 100
        obs[57+7] = self.miss_action
        obs[57+8] = self.state.frame.resources[0].ore / 1000
        obs[57+9] = self.state.frame.resources[0].gas / 1000

        self.miss_action = 0
        return obs

    def _check_done(self):
        return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

    def end_funct(self):
        # default = -540
        time = self.state.frame_from_bwapi / 100
        # t = default + time
        # return (math.sqrt(-5*t) / 10) + 0.8

        return (-7 * ((1 / 50 * time) ** 2)+800) / 10


    def luck_funct(self, time):
        if time < 100:
            return time * 1.0
        elif time >= 100 and time < 200:
            return time * 0.8
        elif time >= 200 and time < 300:
            return time * 0.6
        elif time >= 300 and time < 400:
            return time * 0.4
        elif time >= 400 and time < 500:
            return time * 0.2
        elif time >= 500:
            return time * 0.05

    def life_funct(self, time):
        if time < 100:
            return time * 1.0
        elif time >= 100 and time < 200:
            return time * 0.8
        elif time >= 200 and time < 300:
            return time * 0.6
        elif time >= 300 and time < 400:
            return time * 0.4
        elif time >= 400 and time < 500:
            return time * 0.2
        elif time >= 500:
            return time * 0.05

    def _compute_reward(self):  #보상 계산
        reward = 0

        # if self.obs_pre[57+1] > self.obs[57+1]:     # 라이프 감소하면 -      # 일단 빼보기. 왜냐하면 라이프가 감소하기시작했다는것은 럭이 엄청 나오지 않는한 게임적으로 살아나기 어려움
        #     diff = self.obs_pre[57+1] = self.obs[57+1]
        #     reward -= self.life_funct(self.obs[57+6]/5) * diff
        if self.obs_pre[57+2] < self.obs[57+2]:           # 럭 증가하면 +
            diff = self.obs[57+2] - self.obs_pre[57+2]
            if diff >= 100:
                reward = 0
            reward += 2 * diff
            # reward += self.luck_funct(self.obs[57+6]/5) * diff
        # if self.obs_pre[57+6] < self.obs[57+6]:          # 오래 버틸수록 + -> start_test에서
        #     reward = 0.00001
        # if self.obs[57+7] < 0:
        #     reward = 20 * self.obs[57+7]
        #     self.miss_action = 0
        if self._check_done() and not bool(self.state.battle_won):      # 일찍끝날수록 큰 패널티
            reward -= self.end_funct()

        if self._check_done() and bool(self.state.battle_won):
            reward += 0
            self.episode_wins += 1
        return reward

    def check_normal_resources(self):
        if self.state.frame.resources[0].ore >= 1000:
            return True
        else:
            return False

    def check_hero_resources(self):
        if self.state.frame.resources[0].ore >= 1000 and self.state.frame.resources[0].gas >= 1000:
            return True
        else:
            return False

    def check_bunker_resources(self):
        if self.state.frame.resources[0].ore >= 1000:
            return True
        else:
            return False

    def check_upgrade_resources(self):
        #if self.state.frame.resources[0].ore >= (self.curr_upgrade * 100):
        if self.state.frame.resources[0].ore >= (self.curr_upgrade * 1000):

            return True
        else:
            return False

    def _get_info(self):
        return {}