from populations.excitatory_population import ExcitatoryPopulation


class InputPopulation(ExcitatoryPopulation):
    def _connect_sigle_cell(self, source_section, cell, loc, syn_num_per_source=1, delay=1, weight=1, random_weight=True, source=None, **kwargs):

        syns_4p, heads = cell.make_spine_with_synapse(source=source_section, number=syn_num_per_source, mod_name="Syn4P",
                                                      weight=weight, rand_weight=random_weight, delay=delay, **cell.params_4p_syn,
                                                      source_loc=loc)
        return syns_4p