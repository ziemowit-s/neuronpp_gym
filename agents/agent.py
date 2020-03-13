import abc
from collections import namedtuple
from typing import List

import numpy as np
from neuronpp.cells.cell import Cell

from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

from populations.motor_population import MotorPopulation

AgentOutput = namedtuple("AgentOutput", "index value")

class Agent:
    def __init__(self, input_cell_num, input_size, output_size, input_max_hz, default_stepsize=20):
        self.reward_syns = []
        self.punish_syns = []
        self.observation_syns = []

        self.input_cell_num = input_cell_num
        self.input_size = input_size
        self.output_size = output_size

        self.max_hz = input_max_hz
        self.default_stepsize = default_stepsize
        self.max_stim_per_stepsize = (default_stepsize * input_max_hz) / 1000

        self.input_cells, self.output_cells = self._build_network(input_cell_num=input_cell_num,
                                                                  input_size=input_size, output_cell_num=output_size)
        self.observation_syns = [c.syns for c in self.input_cells]

        self._make_motor_cells(output_cells=self.output_cells, output_cell_num=output_size)
        self._make_records()

        self.sim = None
        self.warmup = None

    def init(self, init_v=-70, warmup=0, dt=0.1):
        """
        :param init_v:
        :param warmup:
        :param dt:
        :return:
        """
        if self.sim is not None:
            raise RuntimeError("Agent have been previously initialized.")

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

    def step(self, observation=None, reward=None, stepsize=None, output_type="time", sort_func=None):
        """

        :param observation:
        :param reward:
        :param stepsize:
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :param sort_func:
            Optional function which define sorting on list of AgentOutput objects.
        :return:
            list(AgentOutput(index, values))
        """
        if self.sim is None:
            raise RuntimeError("Before step you need to initialize the Agent by calling init() function first.")

        if self.observation_syns is None or len(self.observation_syns) == 0:
            raise LookupError("Input synapses field 'self.observation_syns' is empty, but it must match the size of observation.")

        if observation is not None:
            dim = observation.ndim
            if dim == 1:
                self._make_1d_observation(observation, syns=[s for cell, syns in self.observation_syns for s in syns])
            elif dim == 2:
                self._make_2d_observation(observation, syns=self.observation_syns)
            else:
                raise RuntimeError("Observation can be 1D or 2D, but provided %sD" % dim)
        if reward is not None and reward != 0:
            self.make_reward(reward)

        # Run
        if stepsize is None:
            stepsize = self.default_stepsize
        self.sim.run(stepsize)

        output = self._get_output(output_type)
        if sort_func:
            output = sorted(output, key=sort_func)
        return output

    def make_reward_step(self, reward, stepsize=None):
        self.make_reward(reward)
        if stepsize is None:
            stepsize = self.default_stepsize
        self.sim.run(stepsize)

    def make_reward(self, reward):
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
        outputs = []
        min_time = self.sim.t - self.sim.last_runtime
        for i, o in enumerate(self.motor_cells):
            spikes = np.array([i for i in o.get_spikes() if i >= min_time])

            if output_type == "rate":
                s = len(spikes)
            elif output_type == "first-spike":
                s = spikes[0] if len(spikes) > 0 else -1
            elif output_type == "last-spike":
                s = spikes[-1] if len(spikes) > 0 else -1
            elif output_type == "raw":
                s = spikes
            else:
                raise TypeError("Output type can be only string of: 'rate' or 'time', but provided %s" % output_type)
            outputs.append(AgentOutput(index=i, value=s))

        return outputs

    def _make_1d_observation(self, observation, syns):
        #if len(observation) != len(syns):
        #    raise IndexError("Observation sub-array has len %s and synapse sub-array has len %s. Must be equal in len.")

        for obs, syn in zip(observation, syns):
            if obs > 0:
                stim_num, interval = self._get_poisson_stim(obs)
                next_event = 0
                for e in range(stim_num):
                    syn.make_event(next_event)
                    next_event += interval

    def _make_2d_observation(self, observation, syns, x_stride: int = None, y_stride: int = None):
        """

        :param observation:
        :param syns:
        :param x_stride:
            If None - will be of x_window size
        :param y_stride:
            If None - will be of y_window size
        :return:
        """
        x_shape = observation.shape[1]
        y_shape = observation.shape[0]
        x_pixel_size, y_pixel_size = self.get_input_cell_observation_shape(observation)
        if x_stride is None:
            x_stride = x_pixel_size
        if y_stride is None:
            y_stride = y_pixel_size

        syn_i = 0
        for y in range(0, y_shape, y_stride):
            for x in range(0, x_shape, x_stride):

                window = observation[y:y + y_pixel_size, x:x + x_pixel_size]
                if np.sum(window) > 0:
                    self._make_1d_observation(observation=window.flatten(), syns=syns[syn_i])
                syn_i += 1

    def _get_poisson_stim(self, single_input_value):
        stim_num = 0
        stim_int = 0
        if single_input_value > 0:
            stim_num = np.random.poisson(self.max_stim_per_stepsize, 1)[0]
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

    def get_input_cell_observation_shape(self, observation):
        div = np.sqrt(len(self.input_cells))
        x_shape = observation.shape[1]
        y_shape = observation.shape[0]

        x_pixel_size = int(np.ceil(x_shape / div))
        y_pixel_size = int(np.ceil(y_shape / div))
        return x_pixel_size, y_pixel_size
