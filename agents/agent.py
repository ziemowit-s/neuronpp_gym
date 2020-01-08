import abc


class Agent:
    def __init__(self, action_space, observation_space, reward_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_space = reward_space

    @abc.abstractmethod
    def step(self, observation=None, reward=None):
        raise NotImplementedError()
