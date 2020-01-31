from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_cell import Ebner2019Cell
from neuronpp.core.populations.population import Population


class InputPopulation(Population):

    def make_cell(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Ebner2019Cell(name, compile_paths="agents/utils/mods/ebner2019")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("dend", diam=8, l=500, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)
        cell.make_default_mechanisms()
        return cell

    def make_conn(self, cell: Ebner2019Cell, source, source_loc=None, syn_num_per_source=1, delay=1, weight=1,
                  random_weight=True, **kwargs):

        syns_4p, heads = cell.make_spine_with_synapse(source=source, number=syn_num_per_source, mod_name="Syn4P",
                                                      weight=weight, rand_weight=random_weight, delay=delay,
                                                      source_loc=source_loc, **kwargs)
        return syns_4p