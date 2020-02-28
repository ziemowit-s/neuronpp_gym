import numpy as np

from neuronpp.cells.cell import Cell
from neuronpp.utils.record import Record

from agents.agent import Agent
from populations.sigma3_hebbian_population import Sigma3HebbianPopulation
from populations.sigma3_modulatory_population import Sigma3ModulatoryPopulation


class OlfactoryAgent(Agent):
    def __init__(self, input_cell_num, input_size, output_size, max_hz, default_stepsize=20, warmup=10):
        """
        :param input_cell_num:
        :param input_size:
        :param output_size:
        :param max_hz:
        :param default_stepsize:
        :param warmup:
        """
        self.hidden_cells = []
        self.inhibitory_cells = []
        super().__init__(input_cell_num=input_cell_num, input_size=input_size, output_size=output_size,
                         max_hz=max_hz, default_stepsize=default_stepsize, warmup=warmup)

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))

        # INPUTS
        input_pop = Sigma3HebbianPopulation("inp")
        input_pop.create(input_cell_num)
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell,
                          delay=1, random_weight_mean=1.0, netcon_weight=0.1, rule='one')
        # HIDDEN
        self.hidden_pop = self._make_modulatory_population("hid", cell_num=12, source=input_pop)
        self.hidden_cells = self.hidden_pop.cells

        # INHIBITORY NFB
        for i in range(4):
            self._make_inhibitory_cells(counter=i, sources=self.hidden_pop.cells[i:i + 3])

        # OUTPUTS
        output_pop = self._make_modulatory_population("out", cell_num=output_cell_num, source=self.hidden_pop)

        return input_pop.cells, output_pop.cells

    def _make_modulatory_population(self, name, cell_num, source=None):
        pop = Sigma3ModulatoryPopulation(name)
        pop.create(cell_num)

        syns = pop.connect(source=source, syn_num_per_source=1,
                           delay=1, random_weight_mean=1.0, netcon_weight=0.01, rule='all')

        # Prepare synapses for reward and punish
        for hebb, ach, da in [s for slist in syns for s in slist]:
            self.reward_syns.append(da)
            self.punish_syns.append(ach)

        return pop

    def _make_inhibitory_cells(self, counter, sources):
        cell = Cell('inh', compile_paths="agents/commons/mods/sigma3syn")
        cell.name = "Inh[%s][%s]" % (cell.name, counter)
        self.inhibitory_cells.append(cell)

        soma = cell.add_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        w = 0.003  # LTP
        for source in sources:
            cell.add_sypanse(source=source.filter_secs('soma')(0.5), netcon_weight=w, seg=soma(0.5), mod_name="ExcSigma3Exp2Syn")
            source.add_sypanse(source=cell.filter_secs('soma')(0.5), netcon_weight=w, seg=soma(0.5), mod_name="Exp2Syn", e=-90)

    def _make_records(self):
        rec0 = [cell.filter_secs("soma")(0.5) for cell in self.input_cells]
        self.rec_in = Record(rec0, variables='v')
        rec1 = [cell.filter_secs("soma")(0.5) for cell in self.output_cells]
        rec2 = [cell.filter_secs("soma")(0.5) for cell in self.motor_cells]
        self.rec_out = Record(rec1 + rec2, variables='v')
