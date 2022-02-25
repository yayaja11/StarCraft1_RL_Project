import gym
import torchcraft as tc
import torchcraft.Constants as tcc

import TorchCraft.starcraft_gym.proto as proto
import TorchCraft.starcraft_gym.gym_utils as utils

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
                ,(0,0,3)]

class StarCraftEnv(gym.Env):
    def __init__(self, server_ip, server_port, speed, frame_skip, self_play, max_episode_steps):
        print(len(STATE_BUNKER))
        self.client = tc.Client()  # torchcrafy_py에선 여기에 ip랑 port 전달
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.client.connect(self.server_ip, self.server_port)
        self.state = self.client.init(micro_battles=True)   # setup state
        self.speed = speed
        self.frame_skip = frame_skip
        self.self_play = self_play
        self.max_episode_steps = max_episode_steps

        self.episodes = 0
        self.episode_wins = 0
        self.episode_steps = 0

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.state = None
        self.obs = None
        self.obs_pre = None

    def __del__(self):
        self.client.close()

    def empty_commands(self):
        # 돈없으면 기다리기
        self.client.send([])
        self.state = self.client.recv()

    def _step(self, action):

        self.episode_steps += 1
        #print('action:', action)
        while self.state.frame_from_bwapi <= 120: # 80 프레임까지는 유닛이 초기화되지 않을 수 있으므로 대기
            self.empty_commands()

        if STATE_BUNKER[action][2] == 0:        # 일반벙커를 지을때 scv가 일도중이면안되고 히드라 스위치도 꺼져있어야하고, 업그레이드 도중이면 안됨 
            if self.scv_working == 0 and self.hydra_switch == 0 and self.upgrading == 0:
                while self.check_bunker_resources() is False:
                    self.empty_commands()
                self.next_action = ['bunker']
                
        elif STATE_BUNKER[action][2] == 1:      # 영웅벙커를 지을때, scv가 일도중이면안되고 히드라 스위치가 켜져있어야하고, 업그레이드 도중이면 안됨
            if self.scv_working == 0 and self.hydra_switch == 1 and self.upgrading == 0:
                while self.check_bunker_resources() is False:
                    self.empty_commands()
                self.next_action = ['bunker']

            elif self.scv_working == 0 and self.hydra_switch == 0 and self.upgrading == 0:  # 영웅벙커를 지을 때 히드라스위치가 꺼져있으면 켜야함. (일도중x 업그레이드도중x)
                while self.check_hero_resources() is False:
                    self.empty_commands()
                self.next_action = ['hydra']

            else:
                self.empty_commands()

        elif STATE_BUNKER[action][2] == 3:      # 업그레이드할 때, scv가 일도중이면안되고, 업그레이드 도중이면안됨
            if self.scv_working == 0 and self.upgrading == 0:
                while self.check_upgrade_resources() == False:      # 돈있는지
                    self.empty_commands()
                self.next_action = ['upgrade']

            else:
                self.empty_commands()

        whattosend = self._make_commands(self.next_action,STATE_BUNKER[action][0], STATE_BUNKER[action][1])# 위에서 조건에 따라 만들어진 action을 명령어로 제작
        self.number_of_action += 1
        self.client.send(whattosend)
        if self.next_action == ['upgrade']:
            upgrade_start = 1
        self.state = self.client.recv()
        for a in self.state.units[0]:
            if a.type == 7 and a.idle == True:
                self.scv_working == 0
            elif a.type ==7 and a.idle == False:
                self.scv_working == 1

            ihydra = a

        print('I send starcraft:', whattosend, self.number_of_action)
        yy = open('C:\starlog\log_action.txt', 'a')
        yy.write(str(STATE_BUNKER[action]))
        yy.write('\n')
        yy.close()
        # scv가 명령을 받아 벙커지으러 감, upgrade, lurker 모두 일정 시간이 요구됨.
        # scv가 idle이 아닌 상태면 명령 x
        # upgrade가 진행되는중이면 명령 x -> 5 frame?정도. action하고 upgrades_level이 바뀔때까지
        # lurker -> 러커 변태도중 히드라 사라짐.  히드라 스위치 켜져있으면 action x
        #
        # for i in self.state.units[0]:  #
        #     if self.number_of_action > 2:
        #         if i.x == STATE_BUNKER[self.action_save][0] + 6 and i.y == STATE_BUNKER[self.action_save][1] + 1:
        #             self.post_action = True
        #             self.number_of_action += 1
        #         else:
        #             self.post_action = False
        #     else:
        #         self.post_action = True
        if STATE_BUNKER[self.action_save][2] == 3:
            if self.state.frame.resources[0].upgrades_level == self.upgrade_pre and upgrade_start == 1:
                self.upgrading = 1
            else:
                self.upgrading = 0

        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        self.obs_pre = self.obs
        self.upgrade_pre = self.state.frame.resources[0].upgrades_level
        return self.obs, reward, done, info

    def _reset(self):

        utils.print_progress(self.episodes, self.episode_wins)
#        print(not self.self_play, self.episode_steps == self.max_episode_steps)
        if not self.self_play and self.episode_steps == self.max_episode_steps:  # 종료
            print('step reach max')
            print([tcc.restart])
            self.client.send([tcc.restart])
            self.state = self.client.recv()
            while not bool(self.client.state.game_ended):
                self.client.send([])
                self.state = self.client.recv()

        self.episodes += 1
        self.episode_steps = 0

        print(self.episode_steps)
        # self.client.send([tcc.restart])

        self.client.close()
        self.client.connect(self.server_ip, self.server_port)
        self.state = self.client.init(micro_battles=True)

        setup = [[tcc.set_speed, self.speed],  # proto concat_Cmd 다 빼고 list로 만들었음
                 [tcc.set_gui, 1],
                 [tcc.set_frameskip, self.frame_skip],
                 [tcc.set_cmd_optim, 1]]

        self.client.send(setup)
        self.state = self.client.recv() # self.state = self.client.state.d 라는 torchcraft py를 변환한 버전
        print('done',self.state.frame.resources)
        self.obs = self._make_observation()
        self.obs_pre = self.obs
        
        # reset할때 초기화할 값들
        self.countdown = 804
        self.stage = 0
        self.over28stage = 0
        self.hero_bunker = 0
        self.unique_bunker = 0
        self.action_save = 0
        self.curr_upgrade = 1
        self.miss_action = 0
        self.post_action = True
        self.bunker_num = 0
        self.hydra_switch = 0
        self.unique_switch = 0
        self.unique_exception = 0
        self.first_bunker = 0
        self.number_of_action = 0
        self.next_action = []

        return self.obs



        def _action_space(self):
            raise NotImplementedError

        def _observation_space(self):
            raise NotImplementedError

        def _make_commands(self, action,x,y):
            raise NotImplementedError

        def _make_observation(self):
            raise NotImplementedError

        def _compute_reward(self):
            raise NotImplementedError

        def _check_done(self):
            return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

        def _get_info(self):
            return {}

