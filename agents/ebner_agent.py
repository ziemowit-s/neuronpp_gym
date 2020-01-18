import numpy as np

from neuronpp.utils.record import Record
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell

from agents.basic_agent import BasicAgent

WEIGHT = 0.0035


class EbnerAgent(BasicAgent):
    def __init__(self, input_size, max_hz, stepsize=20, finalize_step=5, warmup=200):
        super().__init__(input_size, max_hz, stepsize, finalize_step, warmup)

    def step(self, observation=None, reward=None):
        spiked_pixels = 0
        # Observe
        if observation is not None or reward is not None:
            for input_value, synapse in zip(observation, self.inputs):
                is_spiked = self._make_stim(input_value=input_value, synapse=synapse)
                spiked_pixels += is_spiked

        print('pixel which spiks:', spiked_pixels, 'obs>0:', np.sum(observation > 0))

        # Run
        self.sim.run(self.sim_step + self.finalize_step)

        # Read actions
        output = self.output_rec.recs['v'][0][1].as_numpy()
        return output

    def _prepare_cell(self, input_size, delay=1):
        # Prepare cell
        self.input_cell = Ebner2019AChDACell("input_cell")
        self.input_cell.make_sec("soma", diam=10, l=10, nseg=10)
        self.input_cell.make_sec("dend", diam=4, l=50, nseg=10)
        self.input_cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)

        # make synapses with spines
        syn_4p, heads = self.input_cell.make_spine_with_synapse(source=None, number=100, weight=WEIGHT, mod_name="Syn4PAChDa",
                                                                delay=delay, **self.input_cell.params_4p_syn)
        syn_ach = self.input_cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynACh", sec=heads, delay=delay,
                                                **self.input_cell.params_ach)
        syn_da = self.input_cell.make_sypanses(source=None, weight=WEIGHT, mod_name="SynDa", sec=heads, delay=delay,
                                               **self.input_cell.params_da)
        self.input_cell.set_synaptic_pointers(syn_4p, syn_ach, syn_da)

        # Add mechanisms
        self.input_cell.make_soma_mechanisms()
        self.input_cell.make_apical_mechanisms(sections='dend head neck')

        self.output_rec = Record(self.input_cell.filter_secs(name="soma"), locs=0.5, variables="v")

        return list(zip(syn_4p, syn_ach, syn_da))

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
