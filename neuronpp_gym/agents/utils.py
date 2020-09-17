import numpy as np

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.cells.ebner2019_cell import Ebner2019Cell
from neuronpp.core.distributions import UniformTruncatedDist
from neuronpp.core.populations.population import Population


def make_ebner_network(input_size, input_cell_num):
    def cell_ebner_ach_da():
        cell = Ebner2019AChDACell("ebner_ach_da",
                                  compile_paths="agents/commons/mods/4p_ach_da_syns "
                                                "agents/commons/mods/ebner2019")
        soma = cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
        cell.make_default_mechanisms()
        cell.make_spike_detector(soma(0.5))
        return cell

    def cell_ebner():
        cell = Ebner2019Cell("ebner", compile_paths="agents/commons/mods/ebner2019")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
        cell.make_default_mechanisms()
        return cell

    input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
    inp = Population(name="input")
    inp.add_cells(num=input_cell_num, cell_function=cell_ebner)
    con = inp.connect(rule="one", syn_num_per_cell_source=input_syn_per_cell)
    con.set_source(None)
    con.set_target(inp.cells)
    con.add_synapse("Syn4P").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1))
    con.build()
    
    hid = Population(name="hid")
    hid.add_cells(num=4, cell_function=cell_ebner_ach_da)
    con = hid.connect(syn_num_per_cell_source=1, cell_connection_proba=0.5)
    con.set_source([c.filter_secs("soma")(0.5) for c in inp.cells])
    con.set_target(hid.cells)
    con.add_synapse("Syn4PAChDa").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1)) \
        .add_point_process_params(ACh_tau=50, Da_tau=100, A_Da=1e-35, A_ACh=1e-35)
    con.add_synapse("SynACh").add_netcon(weight=0.3)
    con.add_synapse("SynDa").add_netcon(weight=1.0)
    con.set_synaptic_function(func=lambda syns: Ebner2019AChDACell.set_synaptic_pointers(*syns))
    con.group_synapses()
    con.build()
    reward = [s['SynACh'][0] for s in hid.syns]
    punish = [s['SynDa'][0] for s in hid.syns]
    
    out = Population(name="output")
    out.add_cells(num=2, cell_function=cell_ebner_ach_da)
    con = out.connect(syn_num_per_cell_source=1, cell_connection_proba=1)
    con.set_source([c.filter_secs("soma")(0.5) for c in hid.cells])
    con.set_target(out.cells)
    con.add_synapse("Syn4PAChDa").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1)) \
        .add_point_process_params(ACh_tau=50, Da_tau=100, A_Da=1e-3, A_ACh=1e-3)
    con.add_synapse("SynACh").add_netcon(weight=0.3)
    con.add_synapse("SynDa").add_netcon(weight=1.0)
    con.set_synaptic_function(func=lambda syns: Ebner2019AChDACell.set_synaptic_pointers(*syns))
    con.group_synapses()
    con.build()
    reward += [s['SynACh'][0] for s in out.syns]
    punish += [s['SynDa'][0] for s in out.syns]
    return inp, out, reward, punish


def make_iclamp_cell(input_size, input_cell_num):
    def cell_ebner_ach_da():
        cell = Ebner2019AChDACell("ebner_ach_da",
                                  compile_paths="agents/commons/mods/4p_ach_da_syns "
                                                "agents/commons/mods/ebner2019")
        soma = cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
        cell.make_default_mechanisms()
        cell.make_spike_detector(soma(0.5))
        return cell

    def cell_ebner():
        cell = Ebner2019Cell("ebner", compile_paths="agents/commons/mods/ebner2019")
        cell.add_sec("soma", diam=20, l=20, nseg=10)
        cell.add_sec("apic", diam=2, l=50, nseg=100)
        cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
        cell.make_default_mechanisms()
        return cell

    input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
    inp = Population(name="input")
    inp.add_cells(num=input_cell_num, cell_function=cell_ebner)
    con = inp.connect(rule="one", syn_num_per_cell_source=input_syn_per_cell)
    con.set_source(None)
    con.set_target(inp.cells)
    con.add_synapse("Syn4P").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1))
    con.build()

    out = Population(name="output")
    out.add_cells(num=2, cell_function=cell_ebner_ach_da)
    con = out.connect(syn_num_per_cell_source=1, cell_connection_proba=0.1)
    con.set_source([c.filter_secs("soma")(0.5) for c in inp.cells])
    con.set_target(out.cells)
    con.add_synapse("Syn4PAChDa").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1)) \
        .add_point_process_params(ACh_tau=50, Da_tau=50)
    con.add_synapse("SynACh").add_netcon(weight=0.3)
    con.add_synapse("SynDa").add_netcon(weight=1.0)
    con.set_synaptic_function(func=lambda syns: Ebner2019AChDACell.set_synaptic_pointers(*syns))
    con.group_synapses()
    con.build()
    reward = [s['SynACh'][0] for s in out.syns]
    punish = [s['SynDa'][0] for s in out.syns]
    return inp, out, reward, punish
