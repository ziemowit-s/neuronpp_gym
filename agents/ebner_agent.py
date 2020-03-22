import numpy as np
from neuronpp.utils.record import Record

from agents.agent import Agent
from populations.ebner_hebbian_population import EbnerHebbianPopulation
from populations.ebner_modulatory_population import EbnerModulatoryPopulation

WEIGHT  = 0.0035  # From Ebner et al. 2019
EPSILON = 0.001

class EbnerAgent(Agent):
    def __init__(self, input_cell_num, input_shape, output_size, input_max_hz, default_stepsize=20):
        """
        :param input_cell_num:
        :param input_shape:
        :param output_size:
        :param input_max_hz:
        :param default_stepsize:
        """
        super().__init__(input_cell_num=input_cell_num, input_shape=input_shape, output_size=output_size,
                         input_max_hz=input_max_hz, default_stepsize=default_stepsize)

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
        input_pop = EbnerHebbianPopulation("inp_0")
        input_pop.create(cell_num=input_cell_num)
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell, delay=1,
                          netcon_weight=np.max((0, WEIGHT + np.random.normal(loc=0.0, scale=EPSILON))), rule='one')
        input_pop.add_mechs(single_cell_mechs=self._add_mechs)

        output_pop = self._make_modulatory_population("out_1", cell_num=output_cell_num, source=input_pop)

        return input_pop.cells, output_pop.cells

    # Helper function
    def _add_mechs(self, cell):
        cell.make_soma_mechanisms()
        cell.make_apical_mechanisms(sections='dend head neck')

    def _make_modulatory_population(self, name, cell_num, source=None, syn_per_cell=1):
        pop = EbnerModulatoryPopulation(name)
        pop.create(cell_num)
        pop.add_mechs(single_cell_mechs=self._add_mechs)

        # info draw netcon_weights from N(WEIGHT, epsilon)
        # todo what should be the epsilon value? WEIGHT is small (=0.0035)
        syns = pop.connect(source=source, syn_num_per_source=syn_per_cell,
                           delay=1, netcon_weight=np.max((0, WEIGHT + np.random.normal(loc=0.0, scale=EPSILON))),
                           ach_weight=1, da_weight=1, rule='all', ACh_tau=50, Da_tau=50)

        # Prepare synapses for reward and punish
        for hebb, ach, da in [s for slist in syns for s in slist]:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)

        pop.add_mechs(single_cell_mechs=self._add_mechs)
        return pop

    def _make_records(self):
        rec0 = [cell.filter_secs("soma")(0.5) for cell in self.input_cells]
        self.rec_input = Record(rec0, variables='v')

        rec1 = [cell.filter_secs("soma")(0.5) for cell in self.output_cells]
        self.rec_output = Record(rec1, variables='v')
