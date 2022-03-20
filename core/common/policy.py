class Policy(object):
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def reset_states(self):
        return []

    def on_episode_end(self, episode, logs={}):
        pass

    def get_config(self):

        return {}