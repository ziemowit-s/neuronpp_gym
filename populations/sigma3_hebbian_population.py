from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population


class Sigma3HebbianPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Cell(name, compile_paths="agents/commons/mods/sigma3syn")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.insert('pas')
        cell.insert('hh')
        return cell

    def syn_definition(self, cell: Cell, source, syn_num_per_source=1, delay=1, weight=1, **kwargs):
        secs = cell.filter_secs("apic")
        syns, heads = cell.add_synapses_with_spine(source=source, mod_name="ExcSigma3Exp2Syn", 
                                                   secs=secs, 
                                                   number=syn_num_per_source,
                                                   weight=weight,
                                                   delay=delay, **kwargs)
        return syns