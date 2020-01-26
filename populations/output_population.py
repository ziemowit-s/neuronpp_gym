from populations.excitatory_population import ExcitatoryPopulation


class OutputPopulation(ExcitatoryPopulation):
    def _connect_sigle_cell(self, source_section, cell, loc, syn_num_per_source=1, delay=1, weight=1, random_weight=True, source=None, **kwargs):

        syns_4p, heads = cell.make_spine_with_synapse(source=source_section, number=syn_num_per_source, mod_name="Syn4PAChDa",
                                                      weight=weight, rand_weight=random_weight, delay=delay, **cell.params_4p_syn,
                                                      source_loc=loc)
        # Da is 10x of ACh
        syns_ach = cell.make_sypanses(source=None, weight=weight, mod_name="SynACh", sec=heads, delay=delay)
        syns_da = cell.make_sypanses(source=None, weight=weight * 10, mod_name="SynDa", sec=heads, delay=delay)
        cell.set_synaptic_pointers(syns_4p, syns_ach, syns_da)
        syns = list(zip(syns_4p, syns_ach, syns_da))
        return syns