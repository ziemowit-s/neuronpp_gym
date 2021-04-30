import time
import numpy as np

from neuronpp.utils.graphs.heatmap_graph import HeatmapGraph
from neuronpp.utils.graphs.spikes_heatmap_graph import SpikesHeatmapGraph
from neuronpp_gym.agents.agent2d_da_external import Agent2DDaExternal
from neuronpp_gym.agents.utils import make_hh_network
from neuronpp_gym.core.agent_core import Kernel
from neuronpp_gym.utils import get_env, prepare_pong_observation, reset, get_env_action_number
from collections import deque

DUR = 150
WARMUP = 10
STIM_DUR = 150
STIM_MAX_HZ = 180

TRAIN_NUM = 5000

SCREEN_RATIO = 0.05
AGENT_STEP_AFTER = 2  # env steps
AGENT_STEPSIZE = 40  # in ms

RESET_AFTER = 71  # env steps
REWARD_IF_PASS = 70  # env steps
rewards = deque(maxlen=10)


if __name__ == '__main__':
    # set random seed
    #np.random.seed(31)

    env, obs = get_env('Pong-v0', ratio=SCREEN_RATIO)
    input_shape = obs.shape
    print('input shape', input_shape)

    agent = Agent2DDaExternal(input_max_hz=100, default_stepsize=AGENT_STEPSIZE, tau=100,
                              alpha=0.001, der_avg_num=30)
    agent.build(input_shape=input_shape,
                x_kernel=Kernel(size=3, padding=0, stride=3),  # 18
                y_kernel=Kernel(size=4, padding=0, stride=4))  # 24

    inp_cells, reward_cell, punish_cell, reward_input_syns, punish_input_syns = \
        make_hh_network(input_size=agent.input_size)

    agent.init(init_v=-70, warmup=20, dt=1, input_cells=inp_cells, output_cells=inp_cells,
               reward_cell=reward_cell, punish_cell=punish_cell)
    print("Input neurons:", agent.input_cell_num)

    syn_heatmap0 = HeatmapGraph(name="STILL", elements=inp_cells[0].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)
    #syn_heatmap1 = HeatmapGraph(name="UP", elements=inp_cells[1].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)
    #syn_heatmap2 = HeatmapGraph(name="DOWN", elements=inp_cells[2].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)

    move = -1
    last_agent_steptime = 0
    env_step = 0
    while True:
        env.render()

        action = get_env_action_number(move)
        env_step += 1
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=False)

        if reward != 0 or done or env_step >= RESET_AFTER:
            if env_step >= REWARD_IF_PASS:
                reward = 1
            #print('env step:', env_step)

            reset(env, SCREEN_RATIO)
            env_step = 0
            rewards.append(reward)

        # Make observation
        move = -1
        if env_step % AGENT_STEP_AFTER == 0:
            outputs = agent.step(observation=obs, output_type="rate", poisson=False)

            # Update graphs
            syn_heatmap0.plot()
            #syn_heatmap1.plot()
            #syn_heatmap2.plot()

            # moves from: -1 to 1
            move_vals = [o.value for o in outputs]
            move = np.argmax([o.value for o in outputs])-1

            # write time after agent step
            last_agent_steptime = time.time()

        if reward != 0:
            print('reward:', reward, 'avg_reward:', round(np.average(rewards),2),
                  'move_vals', move_vals)
            agent.reward_step(reward=reward, stepsize=10)

        # make visuatization of mV on each cells by layers
        # agent.rec_input.plot(animate=True, position=(6, 6))
        # agent.rec_output.plot(animate=True)
