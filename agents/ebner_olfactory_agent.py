import numpy as np

from neuronpp.cells.cell import Cell

from agents.ebner_agent import EbnerAgent
from populations.ebner_hebbian_population import EbnerHebbianPopulation

WEIGHT = 0.0035  # From Ebner et al. 2019


class EbnerOlfactoryAgent(EbnerAgent):
    def __init__(self, output_cell_num, input_max_hz, default_stepsize=20):
        """
        :param output_cell_num:
        :param input_max_hz:
        :param default_stepsize:
        """
        self.hidden_cells = []
        self.inhibitory_cells = []
        super().__init__(output_cell_num=output_cell_num, input_max_hz=input_max_hz, default_stepsize=default_stepsize)

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        # INPUTS
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
        input_pop = EbnerHebbianPopulation("inp_0")
        input_pop.create(cell_num=input_cell_num)
        input_pop.connect(source=None, syn_num_per_cell_source=input_syn_per_cell, delay=1, netcon_weight=WEIGHT, rule='one')

        # HIDDEN
        self.hidden_pop = self._make_modulatory_population("hid_1", cell_num=12, source=input_pop)
        self.hidden_cells = self.hidden_pop.cells

        # INHIBITORY NFB
        for i in range(4):
            self._make_inhibitory_cells(population_num=2, counter=i, sources=self.hidden_pop.cells[i:i + 3], netcon_weight=0.1)

        # OUTPUTS
        output_pop = self._make_modulatory_population("out_3", cell_num=output_cell_num, source=self.hidden_pop)

        return input_pop.cells, output_pop.cells

    def _make_inhibitory_cells(self, population_num, counter, netcon_weight, sources):
        cell = Cell('inh', compile_paths="agents/commons/mods/sigma3syn")
        cell.name = "Inh_%s[%s][%s]" % (population_num, cell.name, counter)
        self.inhibitory_cells.append(cell)

        soma = cell.add_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        for source in sources:
            cell.add_synapse(source=source.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                             mod_name="ExcSigma3Exp2Syn")
            source.add_synapse(source=cell.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                               mod_name="Exp2Syn", e=-90)
