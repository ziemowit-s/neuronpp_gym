from typing import List

import numpy as np
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.point_process import PointProcess
from neuronpp.core.populations.population import Population
from neuronpp.utils.utils import set_random_normal_weights


class Sigma3NeuromodulatoryPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        name = "Sigma3Modulatory_%s" % self.cell_counter
        cell = Cell(name, compile_paths="agents/commons/mods/sigma3syn")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(source="apic", target="soma", source_loc=0, target_loc=1)
        cell.insert('pas')
        cell.insert('hh')
        return cell

    def syn_definition(self, cell: Cell, source, syn_num_per_source=1, delay=1,
                       netcon_weight=1, ach_weight=1, da_weight=1, ach_tau=10, da_tau=10,
                       random_weight_mean=None, **kwargs):
        secs = cell.filter_secs("apic")
        syns, heads = cell.add_synapses_with_spine(source=source, mod_name="ExcSigma3Exp2SynAchDa",
                                                   secs=secs,
                                                   number=syn_num_per_source,
                                                   netcon_weight=netcon_weight,
                                                   ach_tau=ach_tau,
                                                   da_tau=da_tau,
                                                   delay=delay, **kwargs)
        if random_weight_mean:
            set_random_normal_weights(point_processes=[s.point_process for s in syns], mean=random_weight_mean,
                                      std=random_weight_mean/4)

        ncs_ach = []
        ncs_da = []
        for syn in syns:
            pp = syn.point_process
            ach_netcon = cell.add_netcon(source=None, point_process=pp,
                                         netcon_weight=ach_weight + pp.hoc.ach_substractor, delay=1)
            da_netcon = cell.add_netcon(source=None, point_process=syn.point_process,
                                        netcon_weight=da_weight + pp.hoc.da_substractor, delay=1)
            ncs_ach.append(ach_netcon)
            ncs_da.append(da_netcon)

        syns = list(zip(syns, ncs_ach, ncs_da))
        return syns