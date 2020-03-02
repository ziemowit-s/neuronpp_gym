from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population
from neuronpp.utils.utils import set_random_normal_weights


class Sigma3HebbianPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "Sigma3Hebbian_%s" % self.cell_counter
        cell = Cell(name, compile_paths="agents/commons/mods/sigma3syn")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.insert('pas')
        cell.insert('hh')
        return cell

    def syn_definition(self, cell: Cell, source, syn_num_per_source=1, delay=1, netcon_weight=1, random_weight_mean=None, **kwargs):
        secs = cell.filter_secs("apic")
        syns, heads = cell.add_synapses_with_spine(source=source, mod_name="ExcSigma3Exp2Syn",
                                                   secs=secs,
                                                   number=syn_num_per_source,
                                                   netcon_weight=netcon_weight,
                                                   delay=delay, **kwargs)
        if random_weight_mean:
            set_random_normal_weights(point_processes=[s.point_process for s in syns], mean=random_weight_mean, std=random_weight_mean)
        return syns