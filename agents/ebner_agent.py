import numpy as np
from neuronpp.utils.record import Record

from agents.agent import Agent
from populations.Exp2SynPopulation import Exp2SynPopulation
from populations.ebner_hebbian_population import EbnerHebbianPopulation
from populations.ebner_neuromodulatory_population import EbnerNeuromodulatoryPopulation


# WEIGHT = 0.0035  # From Ebner et al. 2019

class EbnerAgent(Agent):
    def __init__(self, output_cell_num, input_max_hz, netcon_weight=0.01, default_stepsize=20, ach_tau=50, da_tau=50):
        """
        :param output_cell_num:
        :param input_max_hz:
        :param default_stepsize:
        """
        super().__init__(output_cell_num=output_cell_num, input_max_hz=input_max_hz, default_stepsize=default_stepsize)
        self.netcon_weight = netcon_weight
        self.ach_tau = ach_tau
        self.da_tau = da_tau

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        # info input_syn_per_cell should cover the whole input of a kernel (at least)
        input_syn_per_cell = self.x_kernel.size * self.y_kernel.size
        input_pop = Exp2SynPopulation("inp_0")
        input_pop.create(cell_num=input_cell_num)
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell, delay=1, netcon_weight=self.netcon_weight, rule='one')

        output_pop = self._make_modulatory_population("out_1", cell_num=output_cell_num, source=input_pop)
        #output_pop = EbnerHebbianPopulation("out_1")
        #output_pop.create(cell_num=output_cell_num)
        #output_pop.connect(source=input_pop, delay=1, netcon_weight=self.netcon_weight, rule='all')

        return input_pop.cells, output_pop.cells

    def _make_modulatory_population(self, name, cell_num, source=None):
        pop = EbnerNeuromodulatoryPopulation(name)
        pop.create(cell_num)

        syns = pop.connect(source=source,
                           delay=1, netcon_weight=self.netcon_weight, ach_weight=1, da_weight=1, rule='all',
                           ACh_tau=self.ach_tau, Da_tau=self.da_tau, random_weights=True)

        # Prepare synapses for reward and punish
        for hebb, ach, da in [s for slist in syns for s in slist]:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)

        return pop

    def _make_records(self):
        rec0 = [cell.filter_secs("soma")(0.5) for cell in self.input_cells]
        self.rec_input = Record(rec0, variables='v')

        rec1 = [cell.filter_secs("soma")(0.5) for cell in self.output_cells]
        self.rec_output = Record(rec1, variables='v')
