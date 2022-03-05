#from starcraft_gym.core.common.agent import Agent
import numpy as np
from starcraft_gym.core.common.util import *
import keras.backend as Kb

class ReinForceAgent():
    def __init__(self, state_size, action_size, model, discount_factor=0.99, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.model = model

        self.states, self.actions, self.rewards = [], [], []

    def discount_reward(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        G_t = 0
        for t in reversed(range(0, len(rewards))):
            G_t = rewards[t] + G_t * self.discount_factor
            discounted_rewards[t] = G_t
        return discounted_rewards

    def forward(self, observation):
        state = np.reshape(observation, [1, self.state_size]) # 형태변환
        policy = self.model.predict(state)[0]   # 정책예측
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def backward(self, reward, terminal):
        self.rewards.append(reward)
        if not terminal:
            return

        discounted_rewards = np.float32(self.discount_factor(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)

        if np.std(discounted_rewards) == 0:
            self.states, self.actions, self.rewards = [], [], []
            return

        discounted_rewards /= np.std(discounted_rewards)

        self.train_model([self.states, self.actions, discounted_rewards])

        self.state, self.actions, discounted_rewards = [], [] ,[]

    def compile(self, optimizer, metrics=[]):
        action = Kb.placeholder(shape=[None, self.action_size])
        discounted_rewards = Kb.placeholder(shape=[None,])

        action_prob = Kb.sum(action * self.model.output, axis=1)
        cross_entropy = Kb.log(action_prob) * discounted_rewards
        loss = -Kb.sum(cross_entropy)

        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = Kb.function([self.model.input, action, discounted_rewards], [], updates=updates)
        self.train_model = train
        self.compiled = True
        return

    def load_weights(self, file_path):
        self.model.load_weights(file_path)
        return

    def save_weights(self, file_path, overwrite=False):
        self.model.save_weights(file_path, overwrite)
        return



