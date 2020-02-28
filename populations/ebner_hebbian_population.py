from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_cell import Ebner2019Cell
from neuronpp.core.populations.population import Population


class EbnerHebbianPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Ebner2019Cell(name, compile_paths="agents/commons/mods/ebner2019")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.make_default_mechanisms()
        return cell

    def syn_definition(self, cell: Ebner2019Cell, source, syn_num_per_source=1, delay=1, netcon_weight=1, **kwargs):
        secs = cell.filter_secs("apic")
        syns_4p, heads = cell.add_synapses_with_spine(source=source, mod_name="Syn4P", secs=secs, number=syn_num_per_source,
                                                      netcon_weight=netcon_weight, delay=delay, **kwargs)
        return syns_4p