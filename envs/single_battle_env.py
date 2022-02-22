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
# (x축, y축, z) z: 0: 일반벙커, 1: 영웅벙커, 2: 유니크 벙커 3: upgrade 4: pass
STATE_BUNKER = [(100,39,0), (112,39,0),(124,39,0),(136,39,0),(148,39,0)
                                                               ,(156, 47,0)
                ,(100,55,0),(112,55,0),(124,55,0),(136,55,0),         (156, 55,0)
        ,(88,63,0),(100,63,0),(112,63,0),(124,63,0),(136,63,0),         (156, 63,0),(168,63,0)
,(76,71,0)
,(76,79,0),       (100, 79,0),(112,79,0),(124,79,0),(136,79,0),         (156, 79,0),(168,78,0)
,(76,87,0),       (100, 87,0),(112,87,0),(124,87,0),(136,87,0),         (156, 87,0),(168,87,0)
,(76,95,0),       (100, 95,0),(112,95,0),(124,95,0),(136,95,0),         (156, 95,0),(168,95,0)
,(76,103,0),      (100, 103,0),(112,103,0),(124,103,0),(136,103,0),     (156, 103,0),(168,103,0)
,(76,111,0),                                                    (156, 111,0),(168,111,0)
    ,(84,119,0),(96,119,0),(108,119,0),(120,119,0),(132,119,0),(144,119,0),(156,111,0)
                ,
                (100,39,1), (112,39,1),(124,39,1),(136,39,1),(148,39,1)
                                                               ,(156, 47,1)
                ,(100,55,1),(112,55,1),(124,55,1),(136,55,1),         (156, 55,1)
        ,(88,63,1),(100,63,1),(112,63,1),(124,63,1),(136,63,1),         (156, 63,1),(168,63,1)
,(76,71,1)
,(76,79,1),       (100, 79,1),(112,79,1),(124,79,1),(136,79,1),         (156, 79,1),(168,78,1)
,(76,87,1),       (100, 87,1),(112,87,1),(124,87,1),(136,87,1),         (156, 87,1),(168,87,1)
,(76,95,1),       (100, 95,1),(112,95,1),(124,95,1),(136,95,1),         (156, 95,1),(168,95,1)
,(76,103,1),      (100, 103,1),(112,103,1),(124,103,1),(136,103,1),     (156, 103,1),(168,103,1)
,(76,111,1),                                                    (156, 111,1),(168,111,1)
    ,(84,119,1),(96,119,1),(108,119,1),(120,119,1),(132,119,1),(144,119,1),(156,111,1)
                ,
                (100,39,2), (112,39,2),(124,39,2),(136,39,2),(148,39,2)
                                                               ,(156, 47,2)
                ,(100,55,2),(112,55,2),(124,55,2),(136,55,2),         (156, 55,2)
        ,(88,63,2),(100,63,2),(112,63,2),(124,63,2),(136,63,2),         (156, 63,2),(168,63,2)
,(76,71,2)
,(76,79,2),       (100, 79,2),(112,79,2),(124,79,2),(136,79,2),         (156, 79,2),(168,78,2)
,(76,87,2),       (100, 87,2),(112,87,2),(124,87,2),(136,87,2),         (156, 87,2),(168,87,2)
,(76,95,2),       (100, 95,2),(112,95,2),(124,95,2),(136,95,2),         (156, 95,2),(168,95,2)
,(76,103,2),      (100, 103,2),(112,103,2),(124,103,2),(136,103,2),     (156, 103,2),(168,103,2)
,(76,111,2),                                                    (156, 111,2),(168,111,2)
    ,(84,119,2),(96,119,2),(108,119,2),(120,119,2),(132,119,2),(144,119,2),(156,111,2)

                ,(0,0,3), (0,0,4) ]

class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=100, frame_skip=0, self_play = False, max_episode_steps = 2000):
        # speed 60= 1000프레임을 60초동안. 1=1000프레임을 1초동안
        self.speed = speed
        self.countdown = 806
        self.stage = 0
        self.over28stage = 0
        self.hero_bunker = 0
        self.unique_bunker = 0
        self.action_save = 0  # action저장하고 돈쌓일때까지 기다릴때
        self.curr_upgrade = 1
        self.miss_action = 0
        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed, frame_skip, self_play, max_episode_steps)


    def _action_space(self):

        self.number_of_state = len(STATE_BUNKER)  # + 영웅 + 유니크 + 업그레이드

        return spaces.Discrete(self.number_of_state)

    def _observation_space(self):
        obs_low  = [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8]
        obs_high  = [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8]
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands(self, action):
        cmds = []
        if self.state is None or action is None:
            return cmds

        scv = None
        engineeringbay = None
        hydra = None
        larva = None

        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            return cmds

        # print('buildable?:',self.state.buildable_data)
        # 각각의 유닛이 무엇인지 확인하는 부분
        # for ii in self.state.units[0]:
#        print(issubclass(int, self.state))

        # object_methods = [method_name for method_name in dir(self.state.frame.resources[0].upgrades_level)]  # state하위 모든 메쏘드
        # for i in object_methods:
        #     a = [c for c in dir(i)]
        #     print('1', a)
        # print('1',type(self.state.frame.resources[0].upgrades_level))
        # print('2',self.state.frame.resources[0].upgrades_level)

        # print('mineral:',self.state.units[0][0].id)
        # if self.state.frame.resources[0].ore >= 100:


        for a in self.state.units[0]:
            if a.type == 7 and a.idle == True:
                print('@@@@@')
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

        # if action[0] > 0:
        if scv is None:
            return cmds

        if engineeringbay is None:
            return cmds

        if hydra is None and self.state.frame_from_bwapi >= 50:
            return cmds

        if larva is None and self.stage >= 10:
            return cmds

        # if STATE_BUNKER[action][2] == 2 and self.state.frame.resources[0].orc >= 300  and self.state.frame.resources[0].gas >= 200


        print('action:', STATE_BUNKER[action])

        # if STATE_BUNKER[action][2] == 1 or STATE_BUNKER[action][2] == 2 or STATE_BUNKER[action][2] == 0:
        if STATE_BUNKER[action][2] == 0:  # 일반 벙커
            if self.check_normal_resources():
                cmds.append([
                    tcc.command_unit_protected, scv_id,
                    tcc.unitcommandtypes.Build, -1, STATE_BUNKER[action][0], STATE_BUNKER[action][1], tcc.unittypes.Terran_Supply_Depot])
            else:
                self.miss_action = 1
                return cmds


        elif STATE_BUNKER[action][2] == 1:  # 영웅 벙커
            if self.check_hero_resources():  # 히드라를 만들고 시간이끝나서 다음 판으로 넘어갔을 때, 멈추는듯 or 가는길에 업그레이드해서
                cmds.append([
                    tcc.command_unit_protected, hydra_id,
                    tcc.unitcommandtypes.Morph, -1, -1, -1, tcc.unittypes.Zerg_Lurker]) 
                cmds.append([
                    tcc.command_unit_protected, scv_id,
                    tcc.unitcommandtypes.Build, -1, STATE_BUNKER[action][0], STATE_BUNKER[action][1], tcc.unittypes.Terran_Supply_Depot])
            else:
                print('miss action')
                self.miss_action = 1
                return cmds

        elif STATE_BUNKER[action][2] == 2 and self.stage >= 10:  # 유니크 벙커
            if self.check_unique_resources():
                cmds.append([
                    tcc.command_unit_protected, larva_id,
                    tcc.unitcommandtypes.Morph, -1, -1, -1, tcc.unittypes.Zerg_Drone])
                cmds.append([
                    tcc.command_unit_protected, scv_id,
                    tcc.unitcommandtypes.Build, -1, STATE_BUNKER[action][0], STATE_BUNKER[action][1], tcc.unittypes.Terran_Supply_Depot])
            else:
                self.miss_action = 1
                return cmds

        elif STATE_BUNKER[action][2] == 3:
            if self.check_upgrade_resources():
                cmds.append([
                    tcc.command_unit_protected, engineeringbay_id,
                    tcc.unitcommandtypes.Upgrade, -1, -1, -1, tcc.upgradetypes.Terran_Infantry_Weapons
                ])
            else:
                self.miss_action = 1
                return cmds


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
        obs = np.zeros(self.observation_space.shape)
        lifes = 0
        lucks = 0
        if self.state.units == {}:  # 아무 유닛도 아직 없을때 처리
            return obs

        for a in self.state.units[0]:
            # print(a.x, a.y)
            if a.type == 13:
                lifes += 1

            if a.type == 218:
                lucks += 1
        # f = open('C:\starlog\log.txt', 'a')
        self.countdown -= 1
        if self.countdown == 0:
            self.countdown += 1344
            self.stage += 1
            # f.write(str(self.state.frame_from_bwapi))
            # f.write('\n')
            # print('new_stage')
            # f.write('new_stage\n')

        # if self.state.units[1][-1].x == 76:
        #     f.write(str(self.state.frame_from_bwapi))
        #     f.write('\n')
        #     print('appear')
        #     f.write('appear\n')
        #
        # f.close()

        print('frame_from_bwapi:', self.state.frame_from_bwapi)
        print('countdown timer:', int((self.countdown/14) % int(1344/14)))

        obs[0] = lifes  # 라이프
        obs[1] = lucks  # 럭
        obs[2] = self.state.frame.resources[0].ore  # 미네랄
        obs[3] = self.state.frame.resources[0].gas # 가스
        obs[4] = int((self.countdown/14) % int(1344/14))  # countdown
        obs[5] = self.stage  # stage
        obs[6] = self.curr_upgrade


        return obs

    def _check_done(self):
        return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

    def _compute_reward(self):  #보상 계산
        reward = 0

        if self.obs_pre[0] > self.obs[0]:
            reward = -10
        if self.obs_pre[0] == self.obs[0]:
            reward = 1
        if self.obs_pre[1] < self.obs[1]:
            reward = 1
        self.obs[10] = self.miss_action
        if self.obs[10] == 1:
            self.miss_action = 0
            reward = -50
        if self._check_done() and not bool(self.state.battle_won):
            reward = -500
        if self._check_done() and bool(self.state.battle_won):
            reward = 1000
            self.episode_wins += 1



        return reward

    def check_normal_resources(self):
        if self.state.frame.resources[0].ore >= 100:
            return True
        else:
            return False


    def check_hero_resources(self):
        if self.state.frame.resources[0].ore >= 200 and self.state.frame.resources[0].gas >= 100:
            return True
        else:
            return False

    def check_unique_resources(self):
        if self.state.frame.resources[0].ore >= 300:
            return True
        else:
            return False

    def check_upgrade_resources(self):
        if self.state.frame.resources[0].ore >= (self.curr_upgrade * 100):
            self.curr_upgrade += 1
            return True
        else:
            return False



    def _get_info(self):
        return {}