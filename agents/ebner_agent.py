import numpy as np
from neuron import h
from neuronpp.cells.cell import Cell

from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.utils.record import Record
from neuronpp.utils.run_sim import RunSim


class EbnerAgent:
    def __init__(self, input_cell_num, input_size, output_size, max_hz, weight, motor_weight=1.0, stepsize=20, warmup=200, delay=1):
        """

        :param input_cell_num:
        :param input_size:
        :param output_size:
        :param max_hz:
        :param weight:
            weight for input and output real cells
        :param motor_weight:
            weight for dummy motor cell
        :param stepsize:
        :param warmup:
        :param delay:
        """
        self.stepsize = stepsize
        self.max_stim_per_stepsize = (stepsize * max_hz) / 1000
        self.max_hz = max_hz

        self.inputs = []
        self.hiddens = []
        self.outputs = []

        self.reward_syns = []
        self.punish_syns = []
        self.observation_syns = []

        self.all_other_syns = []
        self.motor_output = []
        self._build_network(input_cell_num=input_cell_num, output_cell_num=output_size, input_size=input_size, delay=delay,
                            weight=weight, motor_weight=motor_weight)

        self.warmup = warmup

        # Create time records
        self.time_vec = h.Vector().record(h._ref_t)

        # Create v records
        rec0 = [cell.filter_secs("soma")[0] for cell, syns in self.inputs]
        self.rec_in = Record(rec0, locs=0.5, variables='v')

        rec1 = [cell.filter_secs("soma")[0] for cell, syns in self.outputs]
        rec2 = [cell.filter_secs("soma")[0] for cell in self.motor_output]
        self.rec_out = Record(rec1 + rec2, locs=0.5, variables='v')

        # init and warmup
        self.sim = RunSim(init_v=-70, warmup=warmup)

    def step(self, observation=None, reward=None):
        """
        Return actions as numpy array of time of spikes in ms.
        """
        if observation is not None:
            self._make_observation(observation)
        if reward is not None:
            self._make_reward(reward)

        # Run
        self.sim.run(self.stepsize)

        # Return actions as time of spikes in ms
        return self.get_motor_output_spike_times(as_global_time=False)

    def get_motor_output_spike_times(self, as_global_time=True):
        """

        :param as_global_time:
        :return:
            Spike times of dummy cells representing motor output stimulation which produce action for dummy motors
        """
        moves = []
        for o in self.motor_output:
            times_of_move = o.get_spikes()
            if not as_global_time:
                min_time = self.sim.t - self.sim.last_runtime
                times_of_move = np.array([i for i in times_of_move if i >= min_time])
                #times_of_move -= min_time
                times_of_move -= self.warmup
            moves.append(times_of_move)
        return moves

    def _build_network(self, input_cell_num, output_cell_num, input_size, weight, motor_weight, delay=1):
        # Make input cells
        for i in range(input_cell_num):
            cell = self._make_single_cell()
            syns = self._make_synapse(cell, number=round(input_size / input_cell_num), delay=delay, weight=weight,
                                      with_neuromodulation=False, is_observation=True)
            self._add_mechs(cell)
            self.inputs.append((cell, syns))

        # Make output cells
        for i in range(output_cell_num):
            cell = self._make_single_cell()
            syns = []
            for c, s in self.inputs:
                syn = self._make_synapse(cell, number=4, delay=delay, source=c.filter_secs("soma")[0], source_loc=0.5,
                                         weight=weight)
                syns.append(syn)
            self._add_mechs(cell)
            self.outputs.append((cell, syns))

        for c, s in self.outputs:
            # Create retro syns
            for c2, s2 in self.inputs:
                syn = self._make_synapse(c, number=4, delay=delay, source=c2.filter_secs("soma")[0], source_loc=0.5,
                                         weight=weight)
                self.all_other_syns.append(syn)
                #syn[0][0].point_process.hoc.e = -80

        for c, s in self.outputs:
            # Create inhibitory to between outputs
            for c2, s2 in self.outputs:
                if c == c2:
                    continue
                syn = self._make_synapse(c, number=4, delay=1, source=c2.filter_secs("soma")[0], source_loc=0.5,
                                         weight=1)
                syn[0][0].point_process.hoc.e = -80
                self.all_other_syns.append(syn)

        # Make motor outputs (dummy cells for motor stimulation)
        self._make_motor_output(weight=motor_weight)

    @staticmethod
    def _make_single_cell():
        cell = Ebner2019AChDACell("input_cell",
                                  compile_paths="agents/utils/mods/ebner2019 agents/utils/mods/4p_ach_da_syns")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("dend", diam=8, l=500, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)
        return cell

    @staticmethod
    def _add_mechs(cell):
        # Add mechanisms
        cell.make_soma_mechanisms()
        cell.make_apical_mechanisms(sections='dend head neck')

    def _make_synapse(self, cell, number, delay, weight, random_weight=True, source=None, source_loc=None,
                      with_neuromodulation=True, is_observation=False):
        """

        :param cell:
        :param number:
        :param delay:
        :param weight:
        :param synapse_type:
            'exc' or 'inh'
        :param random_weight:
        :param source:
        :param source_loc:
        :param with_neuromodulation:
        :return:
        """
        if with_neuromodulation:
            mod = "Syn4PAChDa"
        else:
            mod = "Syn4P"
        syn_4p, heads = cell.make_spine_with_synapse(source=source, number=number, mod_name=mod,
                                                     weight=weight, rand_weight=random_weight, delay=delay, **cell.params_4p_syn,
                                                     source_loc=source_loc)
        if is_observation:
            self.observation_syns.extend(syn_4p)

        if with_neuromodulation:
            syn_ach = cell.make_sypanses(source=None, weight=weight*10, mod_name="SynACh", sec=heads, delay=delay)
            syn_da = cell.make_sypanses(source=None, weight=weight*10, mod_name="SynDa", sec=heads, delay=delay)
            cell.set_synaptic_pointers(syn_4p, syn_ach, syn_da)
            syns = list(zip(syn_4p, syn_ach, syn_da))
        else:
            syns = syn_4p
        return syns

    def _make_motor_output(self, weight):
        """
        Make output for agent's motor/muscle
        """
        for i, (cell, syns) in enumerate(self.outputs):
            sec = cell.filter_secs("soma")[0]
            c = Cell("output%s" % i)
            s = c.make_sec("soma", diam=10, l=10, nseg=1)
            c.insert("hh")
            c.insert("pas")
            c.make_sypanses(source=sec, weight=weight, mod_name="ExpSyn", sec=[s], source_loc=0.5, target_loc=0.5, threshold=-20, e=40, tau=8)
            c.make_spike_detector()
            self.motor_output.append(c)

    def _get_poisson_stim(self, single_input_value):
        stim_num = 0
        stim_int = 0
        if single_input_value > 0:
            stim_num = np.random.poisson(self.max_stim_per_stepsize, 1)[0]
            if stim_num > 0:
                stim_int = self.stepsize/stim_num
                print('STIM!')
        return stim_num, stim_int

    def _make_reward(self, reward):
        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    def _make_observation(self, observation):
        for obs, syn in zip(observation, self.observation_syns):
            if obs > 0:
                stim_num, interval = self._get_poisson_stim(obs)
                next_event = 0
                for e in range(stim_num):
                    syn.make_event(next_event)
                    next_event += interval
