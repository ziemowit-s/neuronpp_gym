import random

import numpy as np
from neuron import h
from neuronpp.cells.cell import Cell

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

WEIGHT = 0.035


class EbnerAgent:
    def __init__(self, input_size, output_size, max_hz, stepsize=20, warmup=200, delay=1):
        self.stepsize = stepsize
        self.max_stim_num = 1000 / stepsize
        self.max_hz = max_hz

        self.inputs = []
        self.outputs = []
        self.motor_output = []
        self._build_cells(input_cell_num=4, output_cell_num=output_size, input_size=input_size, delay=delay)

        # Create time records
        self.time_vec = h.Vector().record(h._ref_t)

        # Create v records
        rec1 = [cell.filter_secs("soma")[0] for cell, syns in self.outputs]
        rec2 = [cell.filter_secs("soma")[0] for cell in self.motor_output]
        records = rec1+rec2
        self.rec = Record(records, locs=0.5, variables='v')

        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)

    def step(self, observation=None, reward=None):
        """

        :param observation:
        :param reward:
        :return:
            Return actions as numpy array of time of spikes in ms.
        """

        if observation is not None:
            syn4ps = [ss[0] for cell, syns in self.inputs for ss in syns]
            for obs, syn in zip(observation, syn4ps):
                self._make_stim(input_value=obs, synapse=syn)

        if reward > 0:
            das = [ss[2] for cell, syns in self.inputs for ss in syns] + [ss[2] for cell, syns in self.inputs for ss in syns]
            for s in das:
                s.make_event(1)
        elif reward < 0:
            achs = [ss[1] for cell, syns in self.inputs for ss in syns] + [ss[1] for cell, syns in self.inputs for ss in syns]
            for s in achs:
                s.make_event(1)

        # Run
        self.sim.run(self.stepsize)

        # Return actions as time of spikes in ms
        moves = self.get_time_of_spikes(as_global_time=False)
        return moves

    def get_time_of_spikes(self, as_global_time=True):
        moves = []
        for o in self.motor_output:
            times_of_move = o.get_spikes()
            if not as_global_time:
                min_time = self.sim.t - self.sim.last_runtime
                times_of_move = np.array([i for i in times_of_move if i >= min_time])
                times_of_move -= min_time
            moves.append(times_of_move)
        return moves

    def _build_cells(self, input_cell_num, output_cell_num, input_size, delay=1):
        for i in range(input_cell_num):
            cell = self._make_single_cell()
            syns = self._make_synapse(cell, number=round(input_size / input_cell_num), delay=delay)
            self._add_mechs(cell)
            self.inputs.append((cell, syns))

        for i in range(output_cell_num):
            cell = self._make_single_cell()
            syns = []
            for c, s in self.inputs:
                syn = self._make_synapse(cell, number=1, delay=delay, source=c.filter_secs("soma")[0], source_loc=0.5)
                syns.append(syn)
            self._add_mechs(cell)
            self.outputs.append((cell, syns))

        # Make output
        self._make_motor_output()

    @staticmethod
    def _make_single_cell():
        cell = Ebner2019AChDACell("input_cell", compile_paths="agents/utils/mods/ebner2019 agents/utils/mods/4p_ach_da_syns agents/utils/mods/neuron_commons")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("dend", diam=8, l=500, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)
        return cell

    @staticmethod
    def _add_mechs(cell):
        # Add mechanisms
        cell.make_soma_mechanisms()
        cell.make_apical_mechanisms(sections='dend head neck')

    @staticmethod
    def _make_synapse(cell, number, delay, source=None, source_loc=None):
        # make synapses with spines
        syn_4p, heads = cell.make_spine_with_synapse(source=source, number=number, mod_name="Syn4PAChDa",
                                                     weight=WEIGHT, rand_weight=True, delay=delay, **cell.params_4p_syn,
                                                     source_loc=source_loc)

        syn_ach = cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynACh", sec=heads, delay=delay, **cell.params_ach)
        syn_da = cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynDa", sec=heads, delay=delay, **cell.params_da)
        cell.set_synaptic_pointers(syn_4p, syn_ach, syn_da)
        input_syns = list(zip(syn_4p, syn_ach, syn_da))
        return input_syns

    def _make_motor_output(self):
        """
        Make output for agent's motor/muscle
        """
        for i, (cell, syns) in enumerate(self.outputs):
            sec = cell.filter_secs("soma")[0]
            c = Cell("output%s" % i)
            s = c.make_sec("soma", diam=10, l=10, nseg=1)
            c.insert("hh")
            c.insert("pas")
            c.make_sypanses(source=sec, weight=0.4, mod_name="ExpSyn", sec=[s], source_loc=0.5, target_loc=0.5, threshold=-20, e=40, tau=8)
            c.make_spike_detector()
            self.motor_output.append(c)

    def _make_stim(self, input_value, synapse):
        """
        :param input_value:
        :param synapse:
            tuple(syn4p, synach, synda)
        :return:
            returns is_spiked bool
        """
        stim_num, interval = self._get_single_stim_params(input_value)

        next_event = interval
        for e in range(stim_num):
            synapse.make_event(next_event)
            next_event += interval

        return stim_num > 0

    def _get_single_stim_params(self, input_value):
        stim_num = int(round((input_value * self.max_hz) / self.max_stim_num))
        stim_int = self.stepsize / stim_num if stim_num > 0 else 0
        return stim_num, stim_int
