import numpy as np
from typing import List, Tuple

from avg_methods.avg_point_der import AvgPointDer
from avg_methods.avg_point_der_online import AvgPointDerOnline
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.synapse import Synapse
from neuronpp.utils.record import Record
from neuronpp_gym.agents.agent2d import Agent2D
from neuronpp_gym.core.agent_core import Kernel


class Agent2DDaExternal(Agent2D):
    def __init__(self, input_max_hz, default_stepsize, tau, alpha, der_avg_num):
        super().__init__(input_max_hz=input_max_hz, default_stepsize=default_stepsize)
        self.last_step_ms = 0
        self.tau = tau
        self.alpha = alpha
        self.der_avg_num = der_avg_num

    def reset(self):
        self.sim.reinit()
        self.sim.run(1)
        self.last_step_ms = 0
        self.derivatives = AvgPointDerOnline(num=self.der_avg_num, global_clip=5, clip_type="inner",
                                             tau_ms=self.tau, dur_single_run_ms=None)

    def init(self, input_cells: List[Cell], output_cells: List[Cell],
             reward_syns: List[Synapse] = None, punish_syns: List[Synapse] = None,
             init_v=-70, warmup=0, dt=0.1, reward_cell=None, punish_cell=None):
        """
        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_cells:
            None for all time
        :param output_cells:
            None for all time
        :param init_v:
        :param warmup:
        :param dt:
        """
        if reward_syns is not None or punish_syns is not None:
            raise TypeError("reward_syns and punish_syns must be none for this class.")

        self.derivatives = AvgPointDerOnline(num=self.der_avg_num, global_clip=5, clip_type="inner",
                                             tau_ms=self.tau, dur_single_run_ms=None)
        self.recs_inp_gs = []
        for s in input_cells[0].syns:
            r = Record(s.point_process, variables="g")
            self.recs_inp_gs.append(r)

        self.reward_cell = reward_cell
        self.punish_cell = punish_cell
        reward_soma = reward_cell.filter_secs("soma")
        punish_soma = punish_cell.filter_secs("soma")

        self.recs_rew_gs = []
        for s in self.reward_cell.syns:
            r = Record(s.point_process, variables="g")
            self.recs_rew_gs.append(r)

        self.recs_pun_gs = []
        for s in self.punish_cell.syns:
            r = Record(s.point_process, variables="g")
            self.recs_pun_gs.append(r)

        self.rec_v_rew = Record(reward_soma(0.5), variables="v")
        self.rec_v_pun = Record(punish_soma(0.5), variables="v")
        self.rec_v_inp = Record(input_cells[0].filter_secs("soma")(0.5), variables="v")

        reward_syn = reward_cell.add_synapse(source=None, mod_name="ExpSyn",
                                             seg=reward_soma(0.5),
                                             netcon_weight=0.01, e=0, tau=1)
        punish_syn = punish_cell.add_synapse(source=None, mod_name="ExpSyn",
                                             seg=punish_soma(0.5),
                                             netcon_weight=0.01, e=0, tau=1)

        super().init(input_cells, output_cells, [reward_syn], [punish_syn], init_v, warmup, dt)

    def build(self, input_shape: Tuple[int, int], x_kernel: Kernel, y_kernel: Kernel):
        """
        Build Agent for 2 Dimensional input.

        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()

        :param input_shape:
            tuple of 2D input shape
        :param x_kernel:
            Object of type Kernel
        :param y_kernel:
            Object of type Kernel
        """
        if self._built:
            raise RuntimeError("The Agent have been already built.")

        if not isinstance(input_shape, tuple) or len(input_shape) != 2:
            raise ValueError("2 dim input shape can be only a tuple of size 2.")

        self.x_kernel_num = self.get_kernel_size(w=input_shape[1], f=x_kernel.size,
                                                 p=x_kernel.padding, s=x_kernel.stride)
        self.y_kernel_num = self.get_kernel_size(w=input_shape[0], f=y_kernel.size,
                                                 p=y_kernel.padding, s=y_kernel.stride)

        self.x_kernel = x_kernel
        self.y_kernel = y_kernel

        padded_obs = self.pad_2d_observation(np.zeros(input_shape))
        input_cell_num = 1

        self._build(input_shape=padded_obs.shape, input_size=padded_obs.size,
                    input_cell_num=input_cell_num)

    def step(self, observation: np.array, output_type="time", sort_func=None, poisson=False,
             stepsize=None):
        output = super().step(observation, output_type, sort_func, poisson, stepsize)

        self._save_der(rec_gs=self.recs_inp_gs, rec_v=self.rec_v_inp, name="inp")
        self._save_der(rec_gs=self.recs_rew_gs, rec_v=self.rec_v_rew, name="rew")
        self._save_der(rec_gs=self.recs_pun_gs, rec_v=self.rec_v_pun, name="pun")

        reward_spikes = self.reward_cell.spikes()
        reward_spikes = reward_spikes[reward_spikes > self.last_step_ms]

        for s in reward_spikes:
            self._train(syns=self.input_cells[0].syns, obs=observation, name="inp", reward=1)
            self._train(syns=self.reward_syns, obs=observation, name="rew", reward=1)
            self._train(syns=self.punish_syns, obs=observation, name="pun", reward=-1)

        punish_spikes = self.punish_cell.spikes()
        punish_spikes = punish_spikes[punish_spikes > self.last_step_ms]

        for s in punish_spikes:
            self._train(syns=self.input_cells[0].syns, obs=observation, name="inp", reward=-1)
            self._train(syns=self.reward_syns, obs=observation, name="rew", reward=-1)
            self._train(syns=self.punish_syns, obs=observation, name="pun", reward=1)

        self.last_step_ms = self.sim.t
        return output

    def _make_reward(self, reward):
        if not self._built:
            raise RuntimeError("Before making reward you need to build the Agent by calling "
                               "build() function first.")
        if self.sim is None:
            raise RuntimeError("Before making reward you need to initialize the Agent by calling "
                               "init() function first.")

        if reward > 0:
            for s in self.reward_syns:
                s.make_event(1)
        elif reward < 0:
            for s in self.punish_syns:
                s.make_event(1)

    def _save_der(self, rec_gs, rec_v, name):
        for i, r in enumerate(rec_gs):
            g = np.average(rec_gs[i].as_numpy().get_records_from_time(self.last_step_ms))
            self.derivatives.append(name=f"{name}_g{i}", val=g)

        self.derivatives.append(name=f'{name}_v', val=-np.average(rec_v.as_numpy().get_records_from_time(self.last_step_ms)))

    def _train(self, syns, obs, name, reward):
        for i, (s, o) in enumerate(zip(syns, obs.flatten())):
            weight = s.netcons[0].get_weight()
            soma_v_dir_g = self.derivatives.get_der(y_name=f'{name}_v', x_name=f"{name}_g{i}",
                                                    dur=self.last_step_ms)
            new_weight = reward * soma_v_dir_g * self.alpha
            if new_weight != 0:
                weight -= new_weight
            s.netcons[0].set_weight(weight)

    def _make_observation(self, observation, poisson=False, stepsize=None):
        """
        Make 2D input observation

        :param observation:
            2 dim array of numbers
        :return:
            list of names of stimulated cells in the stimulation order
        """
        observation = self.pad_2d_observation(observation)
        if self.input_size != observation.size:
            raise RuntimeError("Observation must be of same size as self.input_size, which is "
                               "a product of input_shape.")

        last_syn = 0
        for y in range(0, self.input_shape[0], self.y_kernel.stride):
            for x in range(0, self.input_shape[1], self.x_kernel.stride):

                window = observation[y:y + self.y_kernel.size, x:x + self.x_kernel.size]

                syns = self.input_cells[0].syns[last_syn:last_syn+window.size]
                if np.sum(window) > 0:
                    self._make_single_observation(observation=window.flatten(),
                                                  syns=syns, poisson=poisson,
                                                  stepsize=stepsize)
                last_syn += window.size