# env 와 agent간 연결고리. 환경, 행동, 보상에 따라 다른 형태의 요구사항이 필요할 때 여기서 수정함으로써 agent, env 수정없이 적용 가능
# 그전엔 그냥 아무역할없음

class Processor(object):

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)

    def process_observation(self, observation, state_size=None):
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action

    def process_state_batch(self, batch):
        return batch


    @property
    def metrics(self):
        return []

    @property
    def metrics_names(self):
        return []