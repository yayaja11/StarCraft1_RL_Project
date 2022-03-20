from TorchCraft.starcraft_gym.core.common.policy import  *
import numpy as np
from collections import deque

def get_object_config(o):
    if o is None:
        return None

    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config

class NoisePolicy(Policy):
    def __init__(self, random_process, ratio_of_pure_action=1.0):
        super(NoisePolicy, self).__init__()
        assert random_process is not None
        self.random_process = random_process
        self.ratio_of_pure_action = ratio_of_pure_action

    def select_action(self, pure_action):

        noise = self.random_process.sample()
        action_with_noise = pure_action * self.ratio_of_pure_action + noise
        return action_with_noise, pure_action

    def reset_states(self):
        self.random_process.reset_states()

class GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action

class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim ==1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

class LinearAnnealedPolicy(Policy):
    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy dose not have attribte "{}".'.format(attr))

        super(LinearAnnealedPolicy,self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        a = -float(self.value_max - self.value_min) / float(self.nb_steps)
        b = float(self.value_max)
        value = max(self.value_min, a * float(self.agent.step) + b)
        return value

    def select_action(self, **kwargs):
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)        # 얘를 호출할때 언급한 정책 여기선 epsgreedy

    @property
    def metrics_names(self):
        return ['mean_{}'.format(self.attr)]

    @property
    def metrics(self):
        return [getattr(self.inner_policy, self.attr)]

    def get_config(self):

        config = super(LinearAnnealedPolicy, self).get_config()
        config['attr'] = self.attr
        config['value_max'] = self.value_max
        config['value_min'] = self.value_min
        config['value_test'] = self.value_test
        config['nb_steps'] = self.nb_steps
        config['inner_policy'] = get_object_config(self.inner_policy)
        return config

class MA_EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(MA_EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndmim == 3
        q_values = np.squeeze(q_values, axis = 0)
        nb_agents = q_values.shape[0]
        nb_actions =  q_values.shape[1]

        actions = []
        for agent in range(nb_agents):
            if np.random.uniform() <self.eps:
                action = np.random.random_integers(0, nb_actions - 1)
            else:
                action = np.argmax(q_values[agent])
            actions.append(action)
        actions = np.array(actions)
        return actions

    def get_config(self):
        config = super(MA_EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

class MA_GreedyQPolicy(Policy):
    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions =np.argmax(q_values, axis = -1)
        return actions

class MA_BoltzmannQPolicy(Policy):
    def __init__(self, tau=1., clip=(-500.,500.)):
        super(MA_BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.ndim == 3
        q_values = np.squeeze(q_values, axis=0)
        actions = np.apply_along(self.select_action_agent, -1, q_values)
        return actions

    def select_action_agent(self, q_value):
        assert q_value.ndim == 1
        q_value = q_value.astype('float64')
        nb_actions = q_value.shape[0]

        exp_value = np.exp(np.clip(q_value / self.tau, self.clip[0], self.clip[1]))
        probs = exp_value / np.sum(exp_value)
        action = np.random.choice(range(nb_actions), p=probs)
        return action

    def get_config(self):
        config = super(MA_BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config




