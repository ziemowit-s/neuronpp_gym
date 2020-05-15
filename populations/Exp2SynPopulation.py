from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population
from neuronpp.utils.utils import set_random_normal_weights


class Exp2SynPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "Exp2Syn_%s" % self.cell_counter
        cell = Cell(name)
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.insert('pas')
        cell.insert('hh')
        return cell

    def syn_definition(self, cell: Cell, source, syn_num_per_source=1, delay=1, netcon_weight=1, **kwargs):
        secs = cell.filter_secs("apic")
        syns, heads = cell.add_synapses_with_spine(source=source, mod_name="Exp2Syn",
                                                   secs=secs,
                                                   number=syn_num_per_source,
                                                   netcon_weight=netcon_weight,
                                                   delay=delay, **kwargs)
        return syns
