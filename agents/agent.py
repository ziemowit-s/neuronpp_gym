import abc
from collections import namedtuple
from math import ceil, floor
from typing import List

import numpy as np
from neuronpp.cells.cell import Cell

from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

from populations.motor_population import MotorPopulation

AgentOutput = namedtuple("AgentOutput", "cell_name index value")
Basket = namedtuple("Basket", "cell, size")

class Agent:
    def __init__(self, input_cell_num: int, input_shape: tuple, output_size: int, input_max_hz: int, default_stepsize: int = 20):
        """
        :param input_cell_num:
            Number of input cells
        :param input_shape:
            tuple which specify input shape
            1 or 2 dimentional input is accepted, so input_shape must be a tuple of 1 or 2 elements
        :param output_size:
        :param input_max_hz:
        :param default_stepsize:
        """
        if not isinstance(input_shape, tuple) or len(input_shape) == 0 or len(input_shape) > 2:
            raise ValueError("Param input_shape must be a tuple of 1 or 2 elements, but provided: %s" % input_shape)

        self.reward_syns = []
        self.punish_syns = []

        self.input_cell_num = input_cell_num
        self.input_shape = input_shape
        self.input_ndim = len(input_shape)
        self.input_size = np.prod(input_shape)
        self.output_size = output_size

        self.max_hz = input_max_hz
        self.default_stepsize = default_stepsize
        self.max_stim_per_stepsize = (default_stepsize * input_max_hz) / 1000

        self.input_cells, self.output_cells = self._build_network(input_cell_num=input_cell_num,
                                                                  input_size=self.input_size, output_cell_num=output_size)
        self.input_baskets = self._make_input_baskets()
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
            must be of the same shape as self.input_shape (specified in the constructor param input_shape)
        :param reward:
        :param stepsize:
        :param output_type:
            "time": returns time of first spike for each motor cells.
            "rate": returns number of spikes for each motor cells OR -1 if there were no spike for the cell.
            "raw": returns raw array for each motor cell of all spikes in time in ms.
        :param sort_func:
            Optional function which define sorting on list of AgentOutput objects.
        :return:
            list(AgentOutput(index, cell_name, value))
        """
        if self.sim is None:
            raise RuntimeError("Before step you need to initialize the Agent by calling init() function first.")

        if self.input_cells is None or len(self.input_cells) == 0:
            raise LookupError("Method self._build_network() must return tuple(input_cells, output_cells), "
                              "however input_cells were not defined.")

        # Make observation
        if observation is not None:
            if self.input_ndim != observation.ndim:
                raise RuntimeError("Dimention of the oservation is different than input_shape dimention "
                                   "specified while constructing the agent.")
            for i, o in zip(self.input_shape, observation.shape):
                if i != o:
                    raise RuntimeError("Observation and input_shape specified while constructing the agent must be the same")

            if self.input_ndim == 1:
                self._make_1d_observation(observation)
            elif self.input_ndim == 2:
                self._get_2d_observation_baskets(observation)

        # Make reward
        if reward is not None and reward != 0:
            self.make_reward(reward)

        # Run
        if stepsize is None:
            stepsize = self.default_stepsize
        self.sim.run(stepsize)

        # Prepare output
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
                s = len(spikes)
            elif output_type == "time":
                s = spikes[0] if len(spikes) > 0 else -1
            elif output_type == "raw":
                s = spikes
            else:
                raise TypeError("Output type can be only string of: 'rate' or 'time', but provided %s" % output_type)
            outputs.append(AgentOutput(index=i, cell_name=c.name, value=s))

        return outputs

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

    def _get_2d_basket_shape(self, observation):
        div = np.sqrt(len(self.input_cells))
        x_shape = observation.shape[1]
        y_shape = observation.shape[0]

        x_pixel_size = int(np.ceil(x_shape / div))
        y_pixel_size = int(np.ceil(y_shape / div))
        return x_pixel_size, y_pixel_size

    def _get_1d_basket_size(self, observation):
        return ceil(len(observation)/len(self.input_cells))

    def _make_input_baskets(self):
        baskets = []

        if self.input_ndim == 1:
            basket_sizes = self._get_1d_baskets(shape_size=self.input_size, cells_num=self.input_cell_num)
            for i, x in enumerate(basket_sizes):
                baskets.append(Basket(cell=self.input_cells[i], size=x))

        elif self.input_ndim == 2:
            basket_sizes_x = self._get_1d_baskets(shape_size=self.input_shape[0], cells_num=self.input_cell_num)
            basket_sizes_y = self._get_1d_baskets(shape_size=self.input_shape[1], cells_num=self.input_cell_num)
            for i, (x, y) in enumerate(zip(basket_sizes_x, basket_sizes_y)):
                baskets.append(Basket(cell=self.input_cells[i], size=x*y))
        else:
            raise RuntimeError("Input dimention can be only 1 or 2, but provided: %s" % self.input_ndim)

        print(sum([b.size for b in baskets]), np.prod(self.input_shape))
        return baskets

    def _get_1d_baskets(self, shape_size, num):
        inputs_per_cell = shape_size / num
        inputs_per_cell_int = int(floor(inputs_per_cell))
        input_mod = inputs_per_cell % inputs_per_cell_int

        new_cell = 0
        result = []
        for i in range(num):
            size = inputs_per_cell_int

            new_cell += input_mod
            if new_cell >= 1:
                size += 1
                new_cell -= 1

            result.append(size)

        return result