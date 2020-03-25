import numpy as np

from agents.agent import Agent
from populations.Exp2SynPopulation import Exp2SynPopulation
from populations.sigma3_neuromodulatory_population import Sigma3NeuromodulatoryPopulation


class Sigma3Agent(Agent):
    def __init__(self, output_cell_num, input_max_hz, default_stepsize=20):
        """
        :param output_cell_num:
        :param input_max_hz:
        :param default_stepsize:
        """
        self.hidden_cells = []
        self.inhibitory_cells = []
        self.pattern_cells = []
        super().__init__(output_cell_num=output_cell_num, input_max_hz=input_max_hz, default_stepsize=default_stepsize)

    def _build_network(self, input_cell_num, input_size, output_cell_num):

        # INPUTS
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
        input_pop = Exp2SynPopulation("inp_0")
        input_pop.create(input_cell_num)
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell, delay=1, netcon_weight=0.01, rule='one')

        # HIDDEN
        self.hidden_pop = self._make_modulatory_population("hid_1", cell_num=input_cell_num, source=input_pop)
        self.hidden_cells = self.hidden_pop.cells

        # OUTPUTS
        output_pop = self._make_modulatory_population("out_2", cell_num=output_cell_num, source=self.hidden_pop)

        return input_pop.cells, output_pop.cells

    def _make_modulatory_population(self, name, cell_num, source=None):
        pop = Sigma3NeuromodulatoryPopulation(name)
        pop.create(cell_num)
        syns = pop.connect(source=source, syn_num_per_source=1,
                           delay=1, neuromodulatory_weight=0.1,
                           random_weight_mean=1.0, netcon_weight=0.01, rule='all')
        # Prepare synapses for reward and punish
        for hebb, ach, da in [s for slist in syns for s in slist]:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)

        return pop

    def _make_records(self):
        self.rec_input = self.get_records(cells=self.input_cells)
        self.rec_output = self.get_records(cells=self.output_cells)
