import gym
import torchcraft as tc
import torchcraft.Constants as tcc
import time
import bunker_map as bm

bunker_units = [125, 0, 1, 32]

class StarCraftEnv(gym.Env):
    def __init__(self, server_ip, server_port, speed, frame_skip, self_play, max_episode_steps):
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
        self.bunker_build_state = [0] * 57

        self.state = None
        self.obs = None
        self.obs_pre = None

    def __del__(self):
        self.client.close()

    def print_progress(self, episodes, wins):
        print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
            episodes, wins, wins / (episodes + 1E-6)))

    def check_scv_working(self):
        for i in self.state.units[0]:
            if i.type == 7:
                scv = i
                break
        if scv.idle is False or scv.x == 440:
            self.scv_working = 1
            return True

        else:
            self.scv_working = 0
            return False

    def hydra_make(self):
        # print('-------')
        for i in self.state.frame.units[0]:

            if i.type == tcc.unittypes.Zerg_Hydralisk:
                print('make hero bunker')
                return False
        return True

    def check_bunker_build(self, action):   # 지금은 영웅벙커만인데 노멀벙커에도 적용해야함 그 이후에는 지은 벙커들을 기억할때 판단용도로도
        # 몇몇 영웅벙커에 쓰이는 건물들은 인식을 못함. 다른 인식 방법 찾을필요
        # 벙커짓는 도중에 스테이지가 시작해서 버그가걸리는걸 고쳐보려했으나, 조건이 항상 일치하지 않음. 왠만해선 i.being_constructed는 False,  i.constructing는 True라고 뜨는데
        # 가끔 건너뛰는 상황발생. 학습해서 그런상황을 안만들게 하는방법 + 다른 조건을 찾는 방법
        for i in self.state.units[0]:
            if i.type == tcc.unittypes.Terran_Supply_Depot:
                if self.unbuild_count >= 30:
                    self.next_action = ['halt']
                    whattosend = self._make_commands(self.next_action, self.u)
                    self.client.send(whattosend)
                    self.state = self.client.recv()
                    self.done = self._check_done()
                    temp = self.is_done()
                    if temp[2] == True:
                        return temp
                    self.unbuild_count = 0
                    self.fung = 1
                    self.hydra_switch = 1

                if bm.STATE_BUNKER[action][0] + 2 <= i.x <= bm.STATE_BUNKER[action][0] + 8 and bm.STATE_BUNKER[action][1] - 4 <= i.y <= bm.STATE_BUNKER[action][1] + 3: # 건물이 지어지는 위치가 여러 좌표에 겹쳐 있음.
                    if bm.STATE_BUNKER[action][2] == 0:  # 일반벙커
                        self.bunker_build_state[action] = 1
                    elif bm.STATE_BUNKER[action][2] == 1:  # 영웅벙커
                        self.bunker_build_state[action-57] = 2
                    self.unbuild_count = 0
                    return True

            else:
                pass
        return False

    def is_done(self):     # 흐른 시간 측정

        for i in self.state.frame.units[0]:
            if i.type == tcc.unittypes.Terran_Supply_Depot:
                if i.constructing == True:  # 건설중인 건물인가
                    self.unbuild_count += 1
                    if self.unbuild_count >= 30:
                        self.u = i

        self.post_action = self.action

        if self.done is True:
            self.obs = self._make_observation()
            reward = self._compute_reward()
            info = self._get_info()
            self.obs_pre = self.obs

            return self.obs, reward, self.done, info
        else:
            return [1,1,self.done,1]

    def empty_commands(self):
        # 돈없으면 기다리기
        self.client.send([])
        self.state = self.client.recv()
        self.done = self._check_done()
        for i in self.state.frame.units[0]:
            if i.type == tcc.unittypes.Terran_Supply_Depot:
                if i.constructing == True:  # 건설중인 건물인가
                    self.unbuild_count += 1
                    if self.unbuild_count >= 30:
                        self.u = i

        if self.done is True:
            self.obs = self._make_observation()
            reward = self._compute_reward()
            info = self._get_info()
            self.obs_pre = self.obs
            print('is done in empty commands:', self.done)

            return self.obs, reward, self.done, info
        else:
            return [1,1,self.done,1]

    def _step(self, action):
        self.action = action
        # action = self.test.pop()      # 원하는 행동을 실행시키고 싶을 때
        # self.action = action
        self.episode_steps += 1
        for i in self.state.frame.units[0]:
            if i.type == 0:
                self.now_marineDMG = i.groundATK

        print('-------------------------\nstep start here')
        print('now action:', self.number_of_action)
        print(self.state.frame_from_bwapi)

        while self.state.frame_from_bwapi <= 100: # 100 프레임까지는 유닛이 초기화되지 않을 수 있으므로 대기
            temp = self.empty_commands()
            if temp[2] == True:
                return temp

        if bm.STATE_BUNKER[action][2] == 0:
            print('normal:', bm.STATE_BUNKER[action], self.hydra_switch)
            while self.check_bunker_resources() is False:
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp
            while self.check_scv_working() is True:
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp

            self.next_action = ['bunker']
            whattosend_normal = self._make_commands(self.next_action, action)  # 위에서 조건에 따라 만들어진 action을 명령어로 제작
            self.number_of_action += 1
            self.number_of_normal_bunker += 1
            self.hydra_switch = 0
            self.client.send(whattosend_normal)
            count = 0
            while not self.check_bunker_build(action):
                count += 1
                if count >= 1:      # 너무 많은 action을 보내면 성능저하가 있을가능성 있음.
                    self.client.send(whattosend_normal)
                    self.state = self.client.recv()
                    self.done = self._check_done()
                    count = 0
                    temp = self.is_done()
                    if temp[2] == True:
                        return temp
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp

        elif bm.STATE_BUNKER[action][2] == 1:
            print('hero:', bm.STATE_BUNKER[action], self.hydra_switch)
            while self.hydra_switch == 0:
                while self.check_hero_resources() is False:
                    temp = self.empty_commands()
                    if temp[2] == True:
                        return temp

                self.next_action = ['hydra']
                whattosend_hero = self._make_commands(self.next_action, action)  # 위에서 조건에 따라 만들어진 action을 명령어로 제작
                self.number_of_action += 1
                self.client.send(whattosend_hero)
                self.state = self.client.recv()
                self.pre_frame = self.state.frame_from_bwapi
                temp = self.is_done()

                if temp[2] == True:
                    return temp
                while self.pre_frame + 35 >= self.state.frame_from_bwapi:
                    temp = self.empty_commands()
                    if temp[2] == True:
                        return temp
                self.hydra_switch = 1

            while self.check_bunker_resources() is False:
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp
            while self.check_scv_working() is True:
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp

            self.next_action = ['bunker']
            whattosend_hero = self._make_commands(self.next_action, action)  # 위에서 조건에 따라 만들어진 action을 명령어로 제작
            self.hydra_switch = 0
            self.number_of_action += 1
            self.number_of_hero_bunker += 1
            self.client.send(whattosend_hero)
            count = 0
            while not self.check_bunker_build(action):
                count += 1
                if count >= 1:      # 몇 프레임마다 명령 보낼지
                    self.client.send(whattosend_hero)
                    self.state = self.client.recv()
                    self.done = self._check_done()
                    count = 0
                    temp = self.is_done()

                    if temp[2] == True:
                        return temp
                temp = self.empty_commands()

                if temp[2] == True:
                    return temp

            self.pre_frame = self.state.frame_from_bwapi
            while self.pre_frame + 35 >= self.state.frame_from_bwapi:
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp

        elif bm.STATE_BUNKER[action][2] == 3:
            print('upgrade:', bm.STATE_BUNKER[action])
            for i in self.state.frame.units[0]:
                if i.type == 0:
                    self.post_marineDMG = i.groundATK
            while self.check_upgrade_resources() == False:  # 돈있는지
                temp = self.empty_commands()
                if temp[2] == True:
                    return temp

            self.next_action = ['upgrade']
            self.pre_frame2 = self.state.frame_from_bwapi
            self.number_of_action += 1
            while self.post_marineDMG == self.now_marineDMG:
                whattosend = self._make_commands(self.next_action, action)  # 위에서 조건에 따라 만들어진 action을 명령어로 제작
                self.client.send(whattosend)
                self.state = self.client.recv()
                self.done = self._check_done()
                for i in self.state.frame.units[0]:
                    if i.type == 0:
                        self.now_marineDMG = i.groundATK

                temp = self.is_done()
                if temp[2] == True:
                    return temp
            self.curr_upgrade += 1
            upgrade_completed = 'upgrade completed' + str(self.curr_upgrade) + 'and it takes ' + str(self.state.frame_from_bwapi - self.pre_frame2)
            print(upgrade_completed)

            now = time.localtime()
            yy = open('C:\starlog\log_upgrade.txt', 'a')
            yy.write('---------------------------------\n')
            end_data = str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(
                now.tm_min) + str(now.tm_sec) + 'obs: ' + str(list(map(int, self.obs))) + "\n" + upgrade_completed  # 이전 obs
            yy.write(end_data)
            yy.write('\n')
            yy.close()

        else:
            temp = self.empty_commands()
            if temp[2] == True:
                return temp

        while self.check_scv_working() is True and bm.STATE_BUNKER[action][2] != 3:
            temp = self.empty_commands()
            if temp[2] == True:
                return temp

        self.obs = self._make_observation()
        reward = self._compute_reward()
        info = self._get_info()
        self.obs_pre = self.obs
        print('is done:', self.done)

        return self.obs, reward, self.done, info

    def _reset(self):
        self.print_progress(self.episodes, self.episode_wins)
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
        self.client.close()
        self.client.connect(self.server_ip, self.server_port)
        self.state = self.client.init(micro_battles=True)

        setup = [[tcc.set_speed, self.speed],
                 [tcc.set_gui, 1],
                 [tcc.set_frameskip, self.frame_skip],
                 [tcc.set_cmd_optim, 1]]

        self.client.send(setup)
        self.state = self.client.recv()
        self.obs = self._make_observation()
        self.obs_pre = self.obs
        
        self.curr_upgrade = 0
        self.miss_action = 0
        self.post_action = []
        self.hydra_switch = 0
        self.number_of_action = 0
        self.next_action = []
        self.done = False
        self.bunker_build_state = [0] * 57
        self.number_of_normal_bunker = 0
        self.number_of_hero_bunker = 0
        self.test = [116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,116,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,116,116,116,116,25,24]
        self.unbuild_count = 0
        self.u = 0
        self.post_marineDMG = 0
        self.now_marineDMG = 0
        self.pre_frame2 = 0

        return self.obs

        def _action_space(self):
            raise NotImplementedError

        def _observation_space(self):
            raise NotImplementedError

        def _make_commands(self, action, action_num):
            raise NotImplementedError

        def _make_observation(self):
            raise NotImplementedError

        def _compute_reward(self):
            raise NotImplementedError

        def _check_done(self):
            return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

        def _get_info(self):
            return {}

