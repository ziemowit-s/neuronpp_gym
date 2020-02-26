from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.core.populations.population import Population


class EbnerHebbianModulatoryPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Ebner2019AChDACell(name, compile_paths="agents/commons/mods/ebner2019 agents/commons/mods/4p_ach_da_syns")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.make_default_mechanisms()
        return cell

    def syn_definition(self, cell: Ebner2019AChDACell, source, syn_num_per_source=1, delay=1, weight=1, neuromodulatory_weight=0.01,
                       random_weight=True, **kwargs):
        secs = cell.filter_secs("apic")
        syns_4p, heads = cell.add_synapses_with_spine(source=source, mod_name="Syn4PAChDa", secs=secs, number=syn_num_per_source,
                                                      weight=weight, rand_weight=random_weight, delay=delay, **kwargs)
        # Add neuromodulators
        syns_ach = []
        syns_da = []
        for s, h in zip(syns_4p, heads):
            syn_ach = cell.add_sypanse(source=None, mod_name="SynACh", seg=h(1.0), weight=neuromodulatory_weight, delay=1)
            syn_da = cell.add_sypanse(source=None, mod_name="SynDa", seg=h(1.0), weight=0.1, delay=1)
            cell.set_synaptic_pointers(s, syn_ach, syn_da)
            syns_ach.append(syn_ach)
            syns_da.append(syn_da)

        syns = list(zip(syns_4p, syns_ach, syns_da))
        return syns