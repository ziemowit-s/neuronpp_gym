import numpy as np
from neuron import h
import matplotlib.pyplot as plt
from neuron.units import mV
from neuronpp.cells.cell import Cell

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim

WEIGHT = 0.0035


class EbnerAgent:
    def __init__(self, input_size, max_hz, stepsize=20, warmup=200):
        self.stepsize = stepsize
        self.max_stim_num = 1000 / stepsize
        self.max_hz = max_hz
        self.output_cells = []

        self._build_cells(input_size)
        self.time_vec = h.Vector().record(h._ref_t)

        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)

        print("Agent setup done")

    def step(self, observation=None, reward=None):
        """

        :param observation:
        :param reward:
        :return:
            Return actions as numpy array of time of spikes in ms.
        """

        for iss in [self.input_synapses1, self.input_synapses2]:
            # Observe
            if observation is not None or reward is not None:
                for input_value, synapse in zip(observation, iss):
                    self._make_stim(input_value=input_value, synapse=synapse, reward=reward)

        # Run
        self.sim.run(self.stepsize)

        # Return actions as time of spikes in ms
        moves = self.get_time_of_spikes(as_global_time=False)
        return moves

    def get_time_of_spikes(self, as_global_time=True):
        moves = []
        for o in self.output_cells:
            times_of_move = o.get_spikes()
            if not as_global_time:
                min_time = self.sim.t - self.sim.last_runtime
                times_of_move = np.array([i for i in times_of_move if i >= min_time])
                times_of_move -= min_time
            moves.append(times_of_move)
        return moves

    @staticmethod
    def _make_synapse(cell, input_size, delay, source=None, source_loc=None):
        # make synapses with spines
        syn_4p, heads = cell.make_spine_with_synapse(source=source, number=input_size, weight=WEIGHT, mod_name="Syn4PAChDa",
                                                     delay=delay, **cell.params_4p_syn, source_loc=source_loc)

        syn_ach = cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynACh", sec=heads, delay=delay, **cell.params_ach)
        syn_da = cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynDa", sec=heads, delay=delay, **cell.params_da)
        cell.set_synaptic_pointers(syn_4p, syn_ach, syn_da)
        input_syns = list(zip(syn_4p, syn_ach, syn_da))
        return input_syns

    def _build_cells(self, input_size, delay=1):

        def single_cell():
            # Prepare cell
            cell = Ebner2019AChDACell("input_cell")
            cell.make_sec("soma", diam=10, l=10, nseg=10)
            cell.make_sec("dend", diam=4, l=100, nseg=50)
            cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)

            input_syns = self._make_synapse(cell, input_size=input_size, delay=delay, source=None)

            # Add mechanisms
            cell.make_soma_mechanisms()
            cell.make_apical_mechanisms(sections='dend head neck')

            return cell, input_syns

        self.input_cell, self.input_synapses1 = single_cell()
        self.input_cell2, self.input_synapses2 = single_cell()

        # Inhibitory synapses
        self.c1_c2 = self._make_synapse(self.input_cell2, input_size=5, delay=delay, source=self.input_cell.filter_secs("soma")[0],
                                        source_loc=0.5)
        for s in self.c1_c2:
            s[0].e=-80

        self.c2_c1 = self._make_synapse(self.input_cell, input_size=5, delay=delay, source=self.input_cell2.filter_secs("soma")[0],
                                        source_loc=0.5)
        for s in self.c2_c1:
            s[0].e = -80

        # Make output
        self.make_output(self.input_cell.filter_secs("soma") + self.input_cell2.filter_secs("soma"))

    def make_output(self, secs):
        """
        Make output for agent's motor/muscle
        """
        for i, sec in enumerate(secs):
            c = Cell("output%s" % i)
            s = c.make_sec("soma", diam=10, l=10, nseg=1)
            c.insert("hh")
            c.insert("pas")
            c.make_sypanses(source=sec, weight=0.004, mod_name="ExpSyn", sec=[s], source_loc=0.5, target_loc=0.5, threshold=0, e=40, tau=4)
            c.make_spike_detector()
            self.output_cells.append(c)

    def _make_stim(self, input_value, synapse, reward):
        """
        :param input_value:
        :param synapse:
            tuple(syn4p, synach, synda)
        :return:
            returns is_spiked bool
        """
        stim_num, interval = self.get_single_stim(input_value)

        next_event = interval
        for e in range(stim_num):
            syn4p = synapse[0]
            syn4p.make_event(next_event)
            next_event += interval

        synach = synapse[1]
        synda = synapse[2]
        if reward < 0:
            synach.make_event(1)
        if reward > 0:
            synda.make_event(1)

        return stim_num > 0

    def get_single_stim(self, input_value):
        stim_num = int(round((input_value * self.max_hz) / self.max_stim_num))
        stim_int = self.stepsize / stim_num if stim_num > 0 else 0
        return stim_num, stim_int
