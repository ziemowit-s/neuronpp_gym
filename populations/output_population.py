from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.core.populations.population import Population


class OutputPopulation(Population):

    def make_cell(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Ebner2019AChDACell(name, compile_paths="agents/utils/mods/ebner2019 agents/utils/mods/4p_ach_da_syns")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.make_default_mechanisms()
        return cell

    def make_conn(self, cell: Ebner2019AChDACell, source, source_loc=None, syn_num_per_source=1, delay=1, weight=1,
                  neuromodulatory_weight=0.01, random_weight=True, **kwargs):
        syns_4p, heads = cell.make_spine_with_synapse(source=source, number=syn_num_per_source, mod_name="Syn4PAChDa",
                                                      weight=weight, rand_weight=random_weight, delay=delay,
                                                      source_loc=source_loc, **kwargs)
        # Add neuromodulators
        syns_ach = cell.make_sypanses(source=None, weight=neuromodulatory_weight, mod_name="SynACh", target_sec=heads, target_loc=1.0,
                                      delay=delay)
        syns_da = cell.make_sypanses(source=None, weight=neuromodulatory_weight, mod_name="SynDa", target_sec=heads, target_loc=1.0,
                                     delay=delay)
        cell.set_synaptic_pointers(syns_4p, syns_ach, syns_da)
        syns = list(zip(syns_4p, syns_ach, syns_da))
        return syns