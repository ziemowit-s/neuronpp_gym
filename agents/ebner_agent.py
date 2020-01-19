import numpy as np
from neuron import h

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.run_sim import RunSim

WEIGHT = 0.0035


class EbnerAgent:
    def __init__(self, input_size, max_hz, stepsize=20, finalize_step=5, warmup=200):
        self.stepsize = stepsize
        self.max_stim_num = 1000 / stepsize
        self.max_hz = max_hz
        self.finalize_step = finalize_step

        self.input_synapses = self._build_cells(input_size)
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
        spiked_pixels = 0
        # Observe
        if observation is not None or reward is not None:
            for input_value, synapse in zip(observation, self.input_synapses):
                is_spiked = self._make_stim(input_value=input_value, synapse=synapse)
                spiked_pixels += is_spiked

        print('pixel which spiks:', spiked_pixels, 'obs>0:', np.sum(observation > 0))

        # Run
        self.sim.run(self.stepsize + self.finalize_step)

        # Return actions as time of spikes in ms
        time_of_spikes = self.cell.get_spikes()
        return time_of_spikes

    def _build_cells(self, input_size, delay=1):
        # Prepare cell
        self.cell = Ebner2019AChDACell("input_cell")
        self.cell.make_sec("soma", diam=10, l=10, nseg=10)
        self.cell.make_sec("dend", diam=4, l=50, nseg=10)
        self.cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)

        # make synapses with spines
        syn_4p, heads = self.cell.make_spine_with_synapse(source=None, number=input_size, weight=WEIGHT, mod_name="Syn4PAChDa",
                                                          delay=delay, **self.cell.params_4p_syn)

        syn_ach = self.cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynACh", sec=heads, delay=delay, **self.cell.params_ach)
        syn_da = self.cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynDa", sec=heads, delay=delay, **self.cell.params_da)
        self.cell.set_synaptic_pointers(syn_4p, syn_ach, syn_da)

        # Add mechanisms
        self.cell.make_soma_mechanisms()
        self.cell.make_apical_mechanisms(sections='dend head neck')

        self.cell.make_spike_detector(sec="soma", loc=0.5)

        input_synapses = list(zip(syn_4p, syn_ach, syn_da))
        return input_synapses

    def _make_stim(self, input_value, synapse):
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
            synach = synapse[1]
            synda = synapse[2]

            syn4p.make_event(next_event)
            next_event += interval
        return stim_num > 0

    def get_single_stim(self, input_value):
        stim_num = int(round((input_value * self.max_hz) / self.max_stim_num))
        stim_int = self.stepsize / stim_num if stim_num > 0 else 0
        return stim_num, stim_int
