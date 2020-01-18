import abc

import numpy as np
from neuronpp.utils.run_sim import RunSim

from agents.agent import Agent

WEIGHT = 0.0035


class BasicAgent(Agent):
    """
    Neuromodulated agent for GYM-like environment
    """

    def __init__(self, input_size, max_hz, sim_step=20, finalize_step=5, warmup=200):
        super().__init__()

        self.sim_step = sim_step
        self.max_stim_num = 1000 / sim_step
        self.max_hz = max_hz
        self.finalize_step = finalize_step

        self.inputs = self._prepare_cell(input_size)
        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)
        print("Agent setup done")

    @abc.abstractmethod
    def _prepare_cell(self, input_size):
        """
        :param input_size:
        :return:
            self.inputs which are synapses with source None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, observation=None, reward=None):
        raise NotImplementedError()

    def get_single_stim(self, input_value):
        stim_num = int(round((input_value * self.max_hz) / self.max_stim_num))
        stim_int = self.sim_step / stim_num if stim_num > 0 else 0
        return stim_num, stim_int
