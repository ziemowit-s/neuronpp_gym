import numpy as np
from typing import Tuple

from neuronpp_gym.core.agent_core import AgentCore, Kernel


class Agent2D(AgentCore):
    def __init__(self, input_max_hz, default_stepsize):
        super().__init__(input_max_hz=input_max_hz, default_stepsize=default_stepsize)

        self.x_kernel_num = None
        self.y_kernel_num = None

        self.x_kernel = None
        self.y_kernel = None

    def build(self, input_shape: Tuple[int, int], x_kernel: Kernel, y_kernel: Kernel):
        """
        Build Agent for 2 Dimensional input.

        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_shape:
            tuple of 2D input shape
        :param x_kernel:
            Object of type Kernel
        :param y_kernel:
            Object of type Kernel
        """
        if self._built:
            raise RuntimeError("The Agent have been already built.")

        if not isinstance(input_shape, tuple) or len(input_shape) != 2:
            raise ValueError("2 dim input shape can be only a tuple of size 2.")

        self.x_kernel_num = self.get_kernel_size(w=input_shape[1], f=x_kernel.size,
                                                 p=x_kernel.padding, s=x_kernel.stride)
        self.y_kernel_num = self.get_kernel_size(w=input_shape[0], f=y_kernel.size,
                                                 p=y_kernel.padding, s=y_kernel.stride)

        self.x_kernel = x_kernel
        self.y_kernel = y_kernel

        padded_obs = self.pad_2d_observation(np.zeros(input_shape))
        input_cell_num = self.x_kernel_num * self.y_kernel_num

        self._build(input_shape=padded_obs.shape, input_size=padded_obs.size,
                    input_cell_num=input_cell_num)

    def _make_observation(self, observation, poisson=False, stepsize=None):
        """
        Make 2D input observation

        :param observation:
            2 dim array of numbers
        :return:
            list of names of stimulated cells in the stimulation order
        """
        observation = self.pad_2d_observation(observation)
        if self.input_size != observation.size:
            raise RuntimeError("Observation must be of same size as self.input_size, which is "
                               "a product of input_shape.")

        cell_i = 0
        for y in range(0, self.input_shape[0], self.y_kernel.stride):
            for x in range(0, self.input_shape[1], self.x_kernel.stride):

                current_cell = self.input_cells[cell_i]
                window = observation[y:y + self.y_kernel.size, x:x + self.x_kernel.size]

                if np.sum(window) > 0:
                    self._make_single_observation(observation=window.flatten(),
                                                  syns=current_cell.syns, poisson=poisson,
                                                  stepsize=stepsize)
                cell_i += 1

    def pad_2d_observation(self, obs):
        if self.x_kernel is None or self.y_kernel is None:
            raise RuntimeError("Before calling pad_observation() you need to build the Agent by "
                               "calling build() function first.")
        return np.pad(obs, (self.x_kernel.padding, self.y_kernel.padding), 'constant',
                      constant_values=(0, 0))
