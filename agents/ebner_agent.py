from abc import ABC

import numpy as np
from neuron import h
from neuronpp.cells.cell import Cell

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

from agents.population import Population


class ExcitatoryPopulation(Population, ABC):
    def _create_single_cell(self) -> Cell:
        cell = Ebner2019AChDACell("input_cell", compile_paths="agents/utils/mods/ebner2019 agents/utils/mods/4p_ach_da_syns")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("dend", diam=8, l=500, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)
        return cell


class InputPopulation(ExcitatoryPopulation):
    def _connect_sigle_cell(self, source_section, cell, loc, syn_num_per_source=1, delay=1, weight=1, random_weight=True, source=None, **kwargs):

        syns_4p, heads = cell.make_spine_with_synapse(source=source_section, number=syn_num_per_source, mod_name="Syn4P",
                                                      weight=weight, rand_weight=random_weight, delay=delay, **cell.params_4p_syn,
                                                      source_loc=loc)
        return syns_4p


class OutputPopulation(ExcitatoryPopulation):
    def _connect_sigle_cell(self, source_section, cell, loc, syn_num_per_source=1, delay=1, weight=1, random_weight=True, source=None, **kwargs):

        syns_4p, heads = cell.make_spine_with_synapse(source=source_section, number=syn_num_per_source, mod_name="Syn4PAChDa",
                                                      weight=weight, rand_weight=random_weight, delay=delay, **cell.params_4p_syn,
                                                      source_loc=loc)
        # Da is 10x of ACh
        syns_ach = cell.make_sypanses(source=None, weight=weight, mod_name="SynACh", sec=heads, delay=delay)
        syns_da = cell.make_sypanses(source=None, weight=weight * 10, mod_name="SynDa", sec=heads, delay=delay)
        cell.set_synaptic_pointers(syns_4p, syns_ach, syns_da)
        syns = list(zip(syns_4p, syns_ach, syns_da))
        return syns


class MotorPopuation(Population):
    def _create_single_cell(self) -> Cell:
        cell = Cell("output")
        cell.make_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        cell.make_spike_detector()
        return cell

    def _connect_sigle_cell(self, source_section, cell, loc, weight=1, **kwargs) -> list:
        syns = cell.make_sypanses(source=source_section, weight=weight, mod_name="ExpSyn", sec="soma", source_loc=loc,
                                  target_loc=0.5, threshold=10, e=40, tau=3, delay=3)
        return syns


class EbnerAgent:
    def __init__(self, input_cell_num, input_size, output_size, max_hz, stepsize=20, warmup=200):
        """

        :param input_cell_num:
        :param input_size:
        :param output_size:
        :param max_hz:
        :param stepsize:
        :param warmup:
        """
        self.stepsize = stepsize
        self.max_stim_per_stepsize = (stepsize * max_hz) / 1000
        self.max_hz = max_hz

        self.inputs = []
        self.outputs = []
        self.motor_output = []

        self.reward_syns = []
        self.punish_syns = []
        self.observation_syns = []

        self._build_network(input_cell_num=input_cell_num, output_cell_num=output_size, input_size=input_size)

        self.warmup = warmup

        # Create time records
        self.time_vec = h.Vector().record(h._ref_t)

        # Create v records
        rec0 = [cell.filter_secs("soma")[0] for cell in self.inputs]
        self.rec_in = Record(rec0, locs=0.5, variables='v')

        rec1 = [cell.filter_secs("soma")[0] for cell in self.outputs]
        rec2 = [cell.filter_secs("soma")[0] for cell in self.motor_output]
        self.rec_out = Record(rec1 + rec2, locs=0.5, variables='v')

        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)

    def _build_network(self, input_cell_num, output_cell_num, input_size):
        # Helper function
        def add_mechs(cell):
            cell.make_soma_mechanisms()
            cell.make_apical_mechanisms(sections='dend head neck')

        # INPUTS
        input_pop = InputPopulation()
        self.inputs = input_pop.create(input_cell_num)
        self.observation_syns = input_pop.connect(sources=None, syn_num_per_source=round(input_size / input_cell_num), delay=1, weight=0.01,
                                                  random_weight=True, rule='one')
        input_pop.add_mechs(single_cell_mechs=add_mechs)

        # OUTPUTS
        output_pop = OutputPopulation()
        self.outputs = output_pop.create(output_cell_num)
        syns = output_pop.connect(sources=input_pop.cells, random_weight=True, syn_num_per_source=1, delay=1, weight=0.01, rule='all')
        for hebb, ach, da in syns:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)
        output_pop.add_mechs(single_cell_mechs=add_mechs)

        # MOTOR
        motor_pop = MotorPopuation()
        self.motor_output = motor_pop.create(output_cell_num)
        motor_pop.connect(sources=output_pop.cells, weight=0.1, rule='one')

    def step(self, observation=None, reward=None):
        """
        Return actions as numpy array of time of spikes in ms.
        """
        if self.observation_syns is None or len(self.observation_syns) == 0:
            raise LookupError("Input synapses field 'self.observation_syns' is empty, but it must match the size of observation.")

        if observation is not None:
            dim = observation.ndim
            if dim == 1:
                self._make_1d_observation(observation, syns=self.observation_syns)
            elif dim == 2:
                self._make_2d_observation(observation, syns=self.observation_syns)
            else:
                raise RuntimeError("Observation can be 1D or 2D, but provided %sD" % dim)
        if reward is not None and reward != 0:
            self.make_reward(reward)

        # Run
        self.sim.run(self.stepsize)

        # Return actions as time of spikes in ms
        return self.get_motor_output_spike_times(as_global_time=False)

    def get_motor_output_spike_times(self, as_global_time=True):
        """

        :param as_global_time:
        :return:
            Spike times of dummy cells representing motor output stimulation which produce action for dummy motors
        """
        moves = []
        for o in self.motor_output:
            times_of_move = o.get_spikes()
            if not as_global_time:
                min_time = self.sim.t - self.sim.last_runtime
                times_of_move = np.array([i for i in times_of_move if i >= min_time])
                # times_of_move -= min_time
                times_of_move -= self.warmup
            moves.append(times_of_move)
        return moves

    def _make_1d_observation(self, observation, syns):
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
        inp_len = len(self.inputs)
        x_shape = observation.shape[1]
        y_shape = observation.shape[0]

        x_window = int(np.ceil(x_shape / inp_len))
        y_window = int(np.ceil(y_shape / inp_len))
        if x_stride is None:
            x_stride = x_window
        if y_stride is None:
            y_stride = y_window

        syn_last_i = 0
        for x in range(0, x_shape + x_stride, x_stride):
            for y in range(0, y_shape + y_stride, y_stride):
                window = observation[y:y + y_window, x:x + x_window]
                if np.sum(window) > 0:
                    if window.size == 0:
                        continue
                    self._make_1d_observation(observation=window.flatten(), syns=syns[syn_last_i:syn_last_i + window.size])
                syn_last_i += window.size

    def make_reward(self, reward):
        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    def _get_poisson_stim(self, single_input_value):
        stim_num = 0
        stim_int = 0
        if single_input_value > 0:
            stim_num = np.random.poisson(self.max_stim_per_stepsize, 1)[0]
            if stim_num > 0:
                stim_int = self.stepsize / stim_num
        return stim_num, stim_int
