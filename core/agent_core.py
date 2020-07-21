import abc
import math
from typing import List

import numpy as np
from collections import namedtuple

from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.synapse import Synapse
from neuronpp.utils.simulation import Simulation
from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population


AgentOutput = namedtuple("AgentOutput", "cell_name index value")
Kernel = namedtuple("Kernel", "size padding stride")


class AgentCore:
    def __init__(self, input_max_hz, default_stepsize):
        """
        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_max_hz:
        :param default_stepsize:
        """
        self.reward_syns = []
        self.punish_syns = []

        self.input_max_hz = input_max_hz
        self.default_stepsize = default_stepsize
        self.max_input_stim_per_stepsize = (default_stepsize * input_max_hz) / 1000

        if self.max_input_stim_per_stepsize < 1:
            raise ValueError( "Agent's self.max_input_stim_per_stepsize must be > 1, choose "
                              "input_max_hz and stepsize params carefully.")
        print("max_input_stim_per_stepsize:", self.max_input_stim_per_stepsize)

        self.sim = None
        self.warmup = None

        self.input_size = None
        self.input_shape = None
        self.input_cell_num = None

        self.input_cells = None
        self.output_cells = None

        self._built = False

    def init(self, input_cells: List[Cell], output_cells: List[Cell],
             reward_syns: List[Synapse] = (), punish_syns: List[Synapse] = (),
             init_v=-70, warmup=0, dt=0.1):
        """
        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_cells:
            list of input cells
        :param output_cells:
            list of output cells
        :param init_v:
        :param warmup:
        :param dt:
        """
        self.input_cells = input_cells
        self.output_cells = output_cells

        self.reward_syns = reward_syns
        self.punish_syns = punish_syns

        if self.sim is not None:
            raise RuntimeError("Simulation cannot been run before initialization.")

        self.warmup = warmup
        self.sim = Simulation(init_v=init_v, warmup=warmup, dt=dt, warmup_on_create=True)

    def step(self, observation: np.array, output_type="time", sort_func=None, poisson=False,
             stepsize=None):
        """
        :param observation:
            numpy array. 1 or 2 dim are allowed
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for
            the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :param sort_func:
            Optional function which define sorting on list of AgentOutput objects.
        :param poisson:
            if use Poisson distribution for each pixel stimulation. Default is False.
        :param stepsize:
            in ms. If None - it will use self.default_stepsize.
        :return:
            list(AgentOutput(index, cell_name, value))
        """
        # Check agent's built and initialization before step
        if not self._built:
            raise RuntimeError(
                "Before step you need to build() agent and then initialize by calling init() "
                "function first.")

        if self.sim is None:
            raise RuntimeError(
                "Before step you need to initialize the Agent by calling init() function first.")

        if self.input_cells is None or len(self.input_cells) == 0:
            raise LookupError(
                "Method self._build_network() must return tuple(input_cells, output_cells), "
                "however input_cells were not defined.")

        self._make_observation(observation=observation, poisson=poisson, stepsize=stepsize)

        # Run
        self._make_sim(stepsize)

        # Make output
        output = self._get_output(output_type)
        if sort_func:
            output = sorted(output, key=sort_func)
        return output

    def reward_step(self, reward, stepsize=None):
        """
        It allows to sense the reward by the agent for stepsize time.

        :param reward:
            the value of the reward
        :param stepsize:
            in ms. If None - it will use self.default_stepsize.
        """
        self._make_reward(reward)
        self._make_sim(stepsize)

    @staticmethod
    def get_kernel_size(w, f, p, s):
        """
        Naming convention comes from the Convolutional Neural Networks.

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

    @abc.abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_observation(self, observation, poisson=False, stepsize=None):
        raise NotImplementedError()

    def _build(self, input_shape, input_size, input_cell_num):
        """

        :param input_shape:
        :param input_size:
        :param input_cell_num:
        """
        if self._built or self.sim is not None:
            raise RuntimeError("You must first build agent before initialisation and run "
                               "the simulation.")

        self.input_shape = input_shape
        self.input_size = input_size
        self.input_cell_num = input_cell_num
        self._built = True

    def _make_sim(self, stepsize=None):
        if stepsize is None:
            stepsize = self.default_stepsize
        self.sim.run(stepsize)

    def _make_reward(self, reward):
        if not self._built:
            raise RuntimeError("Before making reward you need to build the Agent by calling "
                               "build() function first.")
        if self.sim is None:
            raise RuntimeError("Before making reward you need to initialize the Agent by calling "
                               "init() function first.")

        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    def _get_output(self, output_type):
        """
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for
            the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :return:
            list(AgentOutput(index, value))
        """
        outputs = []
        min_time = self.sim.t - self.sim.current_runtime
        for i, c in enumerate(self.output_cells):
            spikes = np.array([i for i in c.spikes() if i >= min_time])

            if output_type == "rate":
                s = len(spikes) if len(spikes) > 0 else -1
            elif output_type == "time":
                s = spikes[0] if len(spikes) > 0 else -1
            elif output_type == "raw":
                s = spikes
            else:
                raise TypeError("Output type can be only string of: 'rate' or 'time', "
                                "but provided %s" % output_type)
            outputs.append(AgentOutput(index=i, cell_name=c.name, value=s))

        return outputs

    def _make_single_observation(self, observation, syns, poisson, stepsize=None):
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
                stim_num, interval = self._get_stim_values(pixel, poisson, stepsize)
                next_event = 0
                for e in range(stim_num):
                    syn.make_event(next_event)
                    next_event += interval

    def _get_stim_values(self, pixel, poisson=False, stepsize=None):
        """
        Returns number of events and their interval for a single synaptic stimulation

        :param pixel:
            single pixel value
        :param poisson:
            If use poisson distribution, Default is False.
        :param stepsize:
            stepsize for this observation. if default None - it will take self.default_stepsize
        :return:
            tuple(number of events, interval between events)
        """
        stim_num = 0
        stim_int = 0

        if pixel <= 0:
            return stim_num, stim_int

        if stepsize is None:
            stepsize = self.default_stepsize
            max_stim = self.max_input_stim_per_stepsize
        else:
            max_stim = (stepsize * self.input_max_hz) / 1000

        stim_num = int(round(pixel * max_stim))
        if poisson:
            stim_num = np.random.poisson(stim_num, 1)[0]
        if stim_num > 0:
            stim_int = stepsize / stim_num
        return stim_num, stim_int

    def _add_output_spike_detectors(self):
        for oc in self.output_cells:

            if not hasattr(oc, "_spike_detector"):
                raise TypeError("Output cells must be of type NetConCell and have spike detection "
                                "mechanism.")

            if oc._spike_detector is None:
                soma = oc.filter_secs("soma")
                if isinstance(soma, list):
                    raise LookupError("Output cells need to setup spike detector or at least have "
                                      "a single 'soma' section so that spike detection can be "
                                      "implemented automatically.")

                oc.make_spike_detector(soma(0.5))

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
