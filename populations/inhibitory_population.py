from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population


class InhibitoryPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Cell(name)
        cell.add_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        return cell

    def syn_definition(self, cell: Cell, source, syn_num_per_source=1, delay=1, netcon_weight=1, **kwargs):
        secs = cell.filter_secs("soma")
        syns, heads = cell.add_synapses_with_spine(source=source, mod_name="Exp2Syn", secs=secs, number=syn_num_per_source,
                                                   netcon_weight=netcon_weight, delay=delay, **kwargs)
        return syns