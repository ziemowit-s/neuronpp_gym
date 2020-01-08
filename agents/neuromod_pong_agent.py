import numpy as np
from neuron import h
from neuron.units import mV
import matplotlib.pyplot as plt
from neuronpp.cells.core.netstim_cell import NetStimCell

from neuronpp.cells.core.spine_cell import SpineCell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.Record import Record
from neuronpp.utils.run_sim import RunSim

from agents.agent import Agent

h.load_file('stdrun.hoc')


class Ebner2019AChDaSpineCell(Ebner2019AChDACell, SpineCell):
    def __init__(self, name):
        SpineCell.__init__(self, name)
        Ebner2019AChDACell.__init__(self, name)


WEIGHT = 0.0035


class NeuromodPongAgent(Agent):
    def __init__(self, input_size, pyr_synapse, inh_synapse, sim_step=20):
        super().__init__()

        self.synaptic_delay = 1
        self.sim_step = sim_step
        self.max_stim_num = 1000/sim_step
        self.max_hz = 300

        # Prepare cells
        self.retina_cell = self._make_cell(name="ret", spine_num=input_size)
        self.pyramidal_cell = self._make_cell(name="pyr", spine_num=pyr_synapse)
        self.inhibitory_cell = self._make_cell(name="inh", spine_num=inh_synapse)

        # Prepare input stims
        self.stim_cell = NetStimCell("obs")
        self.nss = []
        for i in range(input_size):
            ns = self.stim_cell.add_netstim("ns[%s]" % i, start=0, number=0)
            self.nss.append(ns)
            self.retina_cell.add_netcons(source=ns, weight=WEIGHT, pp_type_name="Syn4PAChDa", sec_names="head[0][0]", delay=self.synaptic_delay)

        # records
        self.retina_rec = Record(self.retina_cell.filter_point_processes(pp_type_name="Syn4PAChDa", sec_names="head[0][0]"), variables="w")

        # init, run and warmup
        h.finitialize(-70 * mV)
        self.sim = RunSim(warmup=200)

    def step(self, observation=None, reward=None):
        print("agent step..")
        for i, o in enumerate(observation):
            ns = self.nss[i]
            # make input stim
            # start, number, interval, noise
            ns.start = self.sim.time+10

            # max 300 Hz
            stim_num = 20  # (o*self.max_hz)/self.max_stim_num
            stim_int = 1  # self.sim_step/stim_num
            ns.number = stim_num
            ns.interval = stim_int
            ns.noise = 0  # 0.2

        self.sim.run(self.sim_step)
        return 0

    def _make_cell(self, name, spine_num=0):
        cell = Ebner2019AChDaSpineCell(name)
        cell.add_sec("soma", diam=10, l=10, nseg=20)
        cell.add_sec("dend", diam=4, l=100, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)

        print('spine adding')
        cell.add_spines(spine_number=spine_num, head_nseg=1, neck_nseg=1, sections='dend')
        print('mech adding')

        cell.add_soma_mechanisms()
        cell.add_apical_mechanisms(sections='dend head neck')
        cell.add_4p_ach_da_synapse(sec_names="head", loc=1)  # add synapse at the top of each spine's head
        return cell
