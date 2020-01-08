import abc

from neuronpp.cells.core.spine_cell import SpineCell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell

from agents.agent import Agent


class Ebner2019AChDaSpineCell(Ebner2019AChDACell, SpineCell):
    def __init__(self, name):
        SpineCell.__init__(self, name)
        Ebner2019AChDACell.__init__(self, name)


class Ebner2019AChDaAgent(Agent):
    def __init__(self, action_space, observation_space, reward_space):
        super().__init__(action_space, observation_space, reward_space)

    def step(self, observation=None, reward=None):
        return self.action_space
