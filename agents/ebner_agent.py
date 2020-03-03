import numpy as np

from neuronpp.utils.record import Record
from agents.agent import Agent
from populations.ebner_modulatory_population import EbnerModulatoryPopulation
from populations.ebner_hebbian_population import EbnerHebbianPopulation

WEIGHT = 0.0035  # From Ebner et al. 2019


class EbnerAgent(Agent):
    def __init__(self, input_cell_num, input_size, output_size, max_hz, default_stepsize=20, warmup=200):
        """
        :param input_cell_num:
        :param input_size:
        :param output_size:
        :param max_hz:
        :param default_stepsize:
        :param warmup:
        """
        super().__init__(input_cell_num=input_cell_num, input_size=input_size, output_size=output_size,
                         max_hz=max_hz, default_stepsize=default_stepsize, warmup=warmup)

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))

        # INPUTS
        input_pop = EbnerHebbianPopulation("inp")
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell, delay=1, netcon_weight=0.01, rule='one')
        input_pop.add_mechs(single_cell_mechs=self._add_mechs)
        # OUTPUTS
        output_pop = self._make_modulatory_population("out", cell_num=output_cell_num, source=input_pop)
        output_pop.add_mechs(single_cell_mechs=self._add_mechs)

        return input_pop.cells, output_pop.cells

    # Helper function
    def _add_mechs(self, cell):
        cell.make_soma_mechanisms()
        cell.make_apical_mechanisms(sections='dend head neck')

    def _make_modulatory_population(self, name, cell_num, source=None):
        pop = EbnerModulatoryPopulation(name)
        self.output_cells = pop.create(cell_num)

        syns = pop.connect(source=source, syn_num_per_source=1,
                           delay=1, netcon_weight=0.01, neuromodulatory_weight=0.1, rule='all')

        # Prepare synapses for reward and punish
        for hebb, ach, da in [s for slist in syns for s in slist]:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)

        return pop

    def _make_records(self):
        rec0 = [cell.filter_secs("soma")(0.5) for cell in self.input_cells]
        self.rec_in = Record(rec0, variables='v')

        rec1 = [cell.filter_secs("soma")(0.5) for cell in self.output_cells]
        rec2 = [cell.filter_secs("soma")(0.5) for cell in self.motor_cells]
        self.rec_out = Record(rec1 + rec2, variables='v')
