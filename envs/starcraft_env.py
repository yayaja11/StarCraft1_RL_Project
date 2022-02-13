import gym
import torchcraft as tc
import torchcraft.Constants as tcc

import TorchCraft.starcraft_gym.proto as proto
import TorchCraft.starcraft_gym.gym_utils as utils


class StarCraftEnv(gym.Env):
    def __init__(self, server_ip, server_port, speed, frame_skip, self_play, max_episode_steps):
        self.client = tc.Client()  # torchcrafy_py에선 여기에 ip랑 port 전달
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self.client.connect(self.server_ip, self.server_port)
        self.state = self.client.init(micro_battles=True)   #  setup state
#        print(self.state.player_info)
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
        print('6')
    def __del__(self):
        self.client.close()

    def _step(self, action):
        self.episode_steps += 1

        self.client.send(self._make_commands(action))
        self.state = self.client.recv()   # torchcraft_py에선 receive
        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        self.obs_pre = self.obs
        print(self.episode_steps,':',self.obs, reward, done, info)
        return self.obs, reward, done, info

    def _reset(self):
        print('7')

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

        self.obs = self._make_observation()
        self.obs_pre = self.obs
        return self.obs



        def _action_space(self):
            raise NotImplementedError

        def _observation_space(self):
            raise NotImplementedError

        def _make_commands(self, action):
            raise NotImplementedError

        def _make_observation(self):
            raise NotImplementedError

        def _compute_reward(self):
            raise NotImplementedError

        def _check_done(self):
            return bool(self.state.game_ended) or self.state.battle_just_ended  # state에 값을 넣는게 torchcraft에서도 맞는지

        def _get_info(self):
            return {}