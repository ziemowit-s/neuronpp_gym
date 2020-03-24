import abc
import math
from collections import namedtuple
from typing import List

import numpy as np
from neuronpp.cells.cell import Cell

from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

from populations.motor_population import MotorPopulation

AgentOutput = namedtuple("AgentOutput", "cell_name index value")
Kernel = namedtuple("Kernel", "size padding stride")


class Agent:
    def __init__(self, output_cell_num, input_max_hz, default_stepsize):
        self.reward_syns = []
        self.punish_syns = []

        self.output_cell_num = output_cell_num

        self.max_hz = input_max_hz
        self.default_stepsize = default_stepsize
        self.max_input_stim_per_stepsize = (default_stepsize * input_max_hz) / 1000
        if self.max_input_stim_per_stepsize <= 0:
            raise ValueError("Agent's self.max_input_stim_per_stepsize must be > 0, choose input_max_hz and stepsize params carefully.")
        print("max_input_stim_per_stepsize:", self.max_input_stim_per_stepsize)

        self.sim = None
        self.warmup = None
        self.input_size = None
        self.input_shape = None
        self.input_cells = None
        self.output_cells = None
        self.input_cell_num = None

        self.x_kernel = None
        self.y_kernel = None

        self._agent_builded = False

    def build(self, input_shape, x_kernel: Kernel = None, y_kernel: Kernel = None):
        """
        Build Agent based. If x_kernel and y_kernel are provided it will assume 2D input shape, otherwise 1D input shape.

        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_shape:
            integer for 1D or tuple for 2D input shape
        :param x_kernel:
            Object of type Kernel. Default is None
        :param y_kernel:
            Object of type Kernel. Default is None
        """
        if x_kernel is None or y_kernel is None:
            self._build_1d(input_shape)
        else:
            self._build_2d(input_shape, x_kernel, y_kernel)

    def _build_1d(self, input_shape: int):
        if not isinstance(input_shape, int):
            raise ValueError("1 dim input shape must be int.")

        self.input_shape = input_shape
        self._build()

    def _build_2d(self, input_shape: tuple, x_kernel: Kernel, y_kernel: Kernel):
        if not isinstance(input_shape, tuple) or len(input_shape) != 2:
            raise ValueError("2 dim input shape can be only a tuple of size 2.")

        self.x_kernel_num = self.get_kernel_size(w=input_shape[1], f=x_kernel.size, p=x_kernel.padding, s=x_kernel.stride)
        self.y_kernel_num = self.get_kernel_size(w=input_shape[0], f=y_kernel.size, p=y_kernel.padding, s=y_kernel.stride)

        self.x_kernel = x_kernel
        self.y_kernel = y_kernel

        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.input_cell_num = self.x_kernel_num * self.y_kernel_num

        self._build()

    def _build(self):
        if self.sim is not None:
            raise RuntimeError("You must first build agent before initialisation.")
        if self.sim is not None:
            raise RuntimeError("Simulation cannot been run before build.")

        self.input_cells, self.output_cells = self._build_network(input_cell_num=self.input_cell_num,
                                                                  input_size=self.input_size, output_cell_num=self.output_cell_num)

        if len(self.input_cells) != self.input_cell_num:
            raise ValueError("Based on Kernel size input_cell_num is %s, however input_cells returned by _build_network() is: %s" %
                             (self.input_cell_num, len(self.input_cells)))

        if len(self.output_cells) != self.output_cell_num:
            raise ValueError("Based on Kernel size output_cell_num is %s, however output_cells returned by _build_network() is: %s" %
                             (self.output_cell_num, len(self.output_cells)))

        self._make_motor_cells(output_cells=self.output_cells, output_cell_num=self.output_cell_num)
        self._make_records()
        self._agent_builded = True

    def init(self, init_v=-70, warmup=0, dt=0.1):
        """
        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param init_v:
        :param warmup:
        :param dt:
        :return:
        """
        if self.sim is not None:
            raise RuntimeError("Simulation cannot been run before initialization.")

        self.warmup = warmup
        self.sim = RunSim(init_v=init_v, warmup=warmup, dt=dt)

    @abc.abstractmethod
    def _build_network(self, input_cell_num, input_size, output_cell_num) -> (List[Cell], List[Cell]):
        """
        :param input_cell_num:
        :param input_size:
        :param output_cell_num:
        :return:
            Must return list of input cells and output cells
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_records(self):
        raise NotImplementedError()

    def step(self, observation, output_type="time", sort_func=None, stepsize=None):
        """
        :param observation:
            numpy array. 1 or 2 dim are allowed
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :param sort_func:
            Optional function which define sorting on list of AgentOutput objects.
        :param stepsize:
            in ms. If None - it will use self.default_stepsize.
        :return:
            list(AgentOutput(index, cell_name, value))
        """
        # Check agent's built and initialization before step
        if not self._agent_builded:
            raise RuntimeError("Before step you need to build() agent and then initialize by calling init() function first.")

        if self.sim is None:
            raise RuntimeError("Before step you need to initialize the Agent by calling init() function first.")

        if self.input_cells is None or len(self.input_cells) == 0:
            raise LookupError("Method self._build_network() must return tuple(input_cells, output_cells), "
                              "however input_cells were not defined.")

        # Make observation
        if self.input_size != observation.size:
            raise RuntimeError("Observation must be of same size as self.input_size, which is a product of input_shape.")
        if observation.ndim == 1:
            self._make_1d_observation(observation)
        elif observation.ndim == 2:
            self._make_2d_observation(observation)
        else:
            raise ValueError("Observation must be a numpy array of dim 2")

        # Run
        self._make_sim(stepsize)

        # Make output
        output = self._get_output(output_type)
        if sort_func:
            output = sorted(output, key=sort_func)
        return output

    def reward_step(self, reward, stepsize=None):
        self._make_reward(reward)
        self._make_sim(stepsize)

    def _make_sim(self, stepsize=None):
        if stepsize is None:
            stepsize = self.default_stepsize
        self.sim.run(stepsize)

    def _make_reward(self, reward):
        if not self._agent_builded:
            raise RuntimeError("Before making reward you need to build the Agent by calling build() function first.")
        if self.sim is None:
            raise RuntimeError("Before making reward you need to initialize the Agent by calling init() function first.")

        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    @property
    def cells(self):
        cells = []
        for value in self.__dict__.values():
            cs = self._get_recursive_cells(obj=value)
            cells.extend(cs)

        result = []
        for c in cells:
            if c not in result:
                result.append(c)
        return result

    def pad_2d_observation(self, obs):
        if not self._agent_builded:
            raise RuntimeError("Before calling pad_observation() you need to build the Agent by calling build() function first.")
        return np.pad(obs, (self.x_kernel.padding, self.y_kernel.padding), 'constant', constant_values=(0, 0))

    @staticmethod
    def get_kernel_size(w, f, p, s):
        """
        :param w:
            image size of one of dimentions
        :param f:
            convolution size of one of dimentions
        :param p:
            padding
        :param s:
            stride
        :return:
            size of kernel for one of dimention
        """
        return math.floor((w - f + 2 * p) / s + 1)

    def _get_recursive_cells(self, obj):
        acc = []
        if isinstance(obj, CoreCell):
            acc.append(obj)
        elif isinstance(obj, Population):
            acc.extend(obj.cells)
        elif isinstance(obj, list):
            for o in obj:
                ac = self._get_recursive_cells(o)
                acc.extend(ac)
        return acc

    def _get_output(self, output_type):
        """
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :return:
            list(AgentOutput(index, value))
        """
        outputs = []
        min_time = self.sim.t - self.sim.last_runtime
        for i, c in enumerate(self.motor_cells):
            spikes = np.array([i for i in c.get_spikes() if i >= min_time])

            if output_type == "rate":
                s = len(spikes) if len(spikes) > 0 else -1
            elif output_type == "time":
                s = spikes[0] if len(spikes) > 0 else -1
            elif output_type == "raw":
                s = spikes
            else:
                raise TypeError("Output type can be only string of: 'rate' or 'time', but provided %s" % output_type)
            outputs.append(AgentOutput(index=i, cell_name=c.name, value=s))

        return outputs

    def _make_2d_observation(self, obs):
        """
        Make 2D input observation

        :param obs:
        :return:
            list of names of stimulated cells in the stimulation order
        """
        obs = self.pad_2d_observation(obs)

        cell_i = 0
        for y in range(0, self.input_shape[0], self.y_kernel.stride):
            for x in range(0, self.input_shape[1], self.x_kernel.stride):

                current_cell = self.input_cells[cell_i]
                window = obs[y:y + self.y_kernel.size, x:x + self.x_kernel.size]

                if np.sum(window) > 0:
                    self._make_single_observation(observation=window.flatten(), syns=current_cell.syns)
                cell_i += 1

    def _make_1d_observation(self, obs):
        """
        Make 1D input observation

        :param obs:
        1 dim array of numbers
        :return:
        """
        input_syns = [s for c in self.input_cells for s in c.syns]
        self._make_single_observation(observation=obs, syns=input_syns)

    def _make_single_observation(self, observation, syns):
        """
        The core observation method which match observation flat array (1D) to the list of synapses.
        observation and syns need to be of the same length.

        :param observation:
            1 dim array of numbers
        :param syns:
            1 d array of synapses
        """
        if len(observation) != len(syns):
            raise ValueError("Single 1D observation or flatten kernel of 2D observation "
                             "must have the same length as provided subgroup of synapses.")

        for pixel, syn in zip(observation, syns):
            if pixel > 0:
                stim_num, interval = self._get_poisson_stim(pixel)
                next_event = 0
                for e in range(stim_num):
                    syn.make_event(next_event)
                    next_event += interval

    def _get_poisson_stim(self, pixel):
        stim_num = 0
        stim_int = 0
        if pixel > 0:
            stim_num = np.random.poisson(pixel * self.max_input_stim_per_stepsize, 1)[0]
            if stim_num > 0:
                stim_int = self.default_stepsize / stim_num
        return stim_num, stim_int

    def _make_spike_detection(self):
        for oc in self.output_cells:

            if not hasattr(oc, "_spike_detector"):
                raise TypeError("Output cells must be of type NetConCell and have spike detection mechanism.")

            if oc._spike_detector is None:
                soma = oc.filter_secs("soma")
                if isinstance(soma, list):
                    raise LookupError("Output cells need to setup spike detector or at least have a single 'soma' section"
                                      "so that spike detection can be implemented automatically.")

                oc.make_spike_detector(soma(0.5))

    def _make_motor_cells(self, output_cells: List[Cell], output_cell_num, netcon_weight=0.1):
        motor_pop = MotorPopulation("mot")
        self.motor_cells = motor_pop.create(output_cell_num)
        motor_pop.connect(source=output_cells, netcon_weight=netcon_weight, rule='one')

    @staticmethod
    def _get_records(cells, variables="v", sec_name="soma", loc=0.5):
        rec_m = [cell.filter_secs(sec_name)(loc) for cell in cells]
        return Record(rec_m, variables=variables)
