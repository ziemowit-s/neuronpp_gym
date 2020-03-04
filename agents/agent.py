import abc
from typing import List

import numpy as np
from neuronpp.cells.cell import Cell

from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population
from neuronpp.utils.run_sim import RunSim
from neuronpp.utils.utils import show_connectivity_graph

from populations.motor_population import MotorPopulation


class Agent:
    def __init__(self, input_cell_num, input_size, output_size, max_hz, default_stepsize=20, warmup=0):
        self.reward_syns = []
        self.punish_syns = []
        self.observation_syns = []

        self.input_cell_num = input_cell_num
        self.input_size = input_size
        self.output_size = output_size

        self.max_hz = max_hz
        self.default_stepsize = default_stepsize
        self.max_stim_per_stepsize = (default_stepsize * max_hz) / 1000
        self.warmup = warmup

        self.input_cells, self.output_cells = self._build_network(input_cell_num=input_cell_num,
                                                                  input_size=input_size, output_cell_num=output_size)
        self.observation_syns = [c.syns for c in self.input_cells]
        self._make_motor_cells(output_cells=self.output_cells, output_cell_num=output_size)
        self._make_records()

        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)

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

    def step(self, observation=None, reward=None, stepsize=None):
        """
        Return actions as numpy array of time of spikes in ms.
        """
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

        # Return actions as time of spikes in ms
        return self._get_motor_output_spike_times(as_global_time=False)

    def make_reward(self, reward):
        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    def show_connectivity_graph(self):
        cells = []
        for v in self.__dict__.values():
            cs = self._get_cells(v)
            cells.extend(cs)
        show_connectivity_graph(cells)
        # return cells
    
    def get_cells(self):
        cells = []
        for v in self.__dict__.values():
            cs = self._get_cells(v)
            cells.extend(cs)
        return list(set(cells))

    def _make_motor_cells(self, output_cells: List[Cell], output_cell_num):
        motor_pop = MotorPopulation("mot")
        self.motor_cells = motor_pop.create(output_cell_num)
        motor_pop.connect(source=output_cells, netcon_weight=0.1, rule='one')

    def _get_motor_output_spike_times(self, as_global_time=True):
        """
        :param as_global_time:
        :return:
            Spike times of dummy cells representing motor output stimulation which produce action for dummy motors
        """
        moves = []
        for o in self.motor_cells:
            times_of_move = o.get_spikes()
            if not as_global_time:
                min_time = self.sim.t - self.sim.last_runtime
                times_of_move = np.array([i for i in times_of_move if i >= min_time])
                # times_of_move -= min_time
                times_of_move -= self.warmup
            moves.append(times_of_move)
        return moves

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
        div = np.sqrt(len(self.input_cells))
        x_shape = observation.shape[1]
        y_shape = observation.shape[0]

        x_window = int(np.ceil(x_shape / div))
        y_window = int(np.ceil(y_shape / div))
        if x_stride is None:
            x_stride = x_window
        if y_stride is None:
            y_stride = y_window

        syn_i = 0
        for y in range(0, y_shape, y_stride):
            for x in range(0, x_shape, x_stride):

                window = observation[y:y + y_window, x:x + x_window]
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

    def _get_cells(self, v):
        acc = []
        if isinstance(v, CoreCell):
            acc.append(v)
        elif isinstance(v, Population):
            acc.extend(v.cells)
        elif isinstance(v, list):
            for vv in v:
                ac = self._get_cells(vv)
                acc.extend(ac)
        return acc
