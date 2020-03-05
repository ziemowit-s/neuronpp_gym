from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.core.populations.population import Population
from neuronpp.utils.utils import set_random_normal_weights


class EbnerModulatoryPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "EbnerHebbianModulatory_%s" % self.cell_counter
        cell = Ebner2019AChDACell(name, compile_paths="agents/commons/mods/ebner2019 agents/commons/mods/4p_ach_da_syns")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.make_default_mechanisms()
        return cell

    def syn_definition(self, cell: Ebner2019AChDACell, source, syn_num_per_source=1, delay=1, netcon_weight=1,
                       random_weight_mean=None, neuromodulatory_weight=0.01, **kwargs):
        secs = cell.filter_secs("apic")
        syns_4p, heads = cell.add_synapses_with_spine(source=source, mod_name="Syn4PAChDa", secs=secs, number=syn_num_per_source,
                                                      netcon_weight=netcon_weight, delay=delay, **kwargs)

        if random_weight_mean:
            set_random_normal_weights(point_processes=[s.point_process for s in syns_4p], mean=random_weight_mean,
                                      std=random_weight_mean)

        # Add neuromodulators
        syns_ach = []
        syns_da = []
        for s, h in zip(syns_4p, heads):
            syn_ach = cell.add_sypanse(source=None, mod_name="SynACh", seg=h(1.0), netcon_weight=neuromodulatory_weight, delay=1)
            syn_da = cell.add_sypanse(source=None, mod_name="SynDa", seg=h(1.0), netcon_weight=neuromodulatory_weight, delay=1)
            cell.set_synaptic_pointers(s, syn_ach, syn_da)
            syns_ach.append(syn_ach)
            syns_da.append(syn_da)

        syns = list(zip(syns_4p, syns_ach, syns_da))
        return syns