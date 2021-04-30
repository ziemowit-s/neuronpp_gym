import time
import numpy as np
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import *
from matplotlib.pyplot import imshow

from neuronpp.utils.graphs.heatmap_graph import HeatmapGraph
from neuronpp.utils.graphs.spikes_heatmap_graph import SpikesHeatmapGraph
from neuronpp_gym.agents.agent2d_da_external import Agent2DDaExternal
from neuronpp_gym.agents.utils import make_hh_network
from neuronpp_gym.core.agent_core import Kernel
from neuronpp_gym.utils import get_env, prepare_pong_observation
from collections import deque

DUR = 150
WARMUP = 10
STIM_DUR = 150
STIM_MAX_HZ = 180

TRAIN_NUM = 5000

SCREEN_RATIO = 0.1
AGENT_STEP_AFTER = 2  # env steps
AGENT_STEPSIZE = 40  # in ms

RESET_AFTER = 71  # env steps
REWARD_IF_PASS = 70  # env steps
rewards = deque(maxlen=10)

def reset():
    obs = np.zeros([5,5])
    obs[0,0] = 1
    obs[-1,-1] = 0.5
    return obs, 0

def step(obs, action, env_step):
    max_ids = np.unravel_index(obs.argmax(), obs.shape)
    obs[max_ids[0], max_ids[1]] = 0
    dist_prev = (max_ids[0] + max_ids[1]) / 8
    reward = 0
    try:
        if action == 0: # UP
            obs[max_ids[0]-1, max_ids[1]] = 1
        elif action == 1: # DOWN
            obs[max_ids[0]+1, max_ids[1]] = 1
        elif action == 2: # right
            obs[max_ids[0], max_ids[1]+1] = 1
        elif action == 3: # left
            obs[max_ids[0], max_ids[1]-1] = 1
        else:
            obs[max_ids[0], max_ids[1]] = 1
    except IndexError:
        obs[max_ids[0], max_ids[1]] = 1
        reward = -1

    done = 0
    max_ids = np.unravel_index(obs.argmax(), obs.shape)
    dist_new = (max_ids[0] + max_ids[1]) / 8
    if reward == 0:
        reward = dist_new - dist_prev

    if reward == 0:
        reward = -1

    if dist_new == 1:
        done = 1
        reward = 1
    if env_step > 25:
        done = 1
        reward = -1

    if done:
        rewards.append(dist_new)
    return obs, reward, done


if __name__ == '__main__':
    # set random seed
    np.random.seed(31)

    obs, env_step = reset()

    input_shape = obs.shape
    print('input shape', input_shape)

    agent = Agent2DDaExternal(input_max_hz=100, default_stepsize=AGENT_STEPSIZE, tau=500,
                              alpha=0.01, der_avg_num=25)
    agent.build(input_shape=input_shape,
                x_kernel=Kernel(size=3, padding=0, stride=3),  # 18
                y_kernel=Kernel(size=4, padding=0, stride=4))  # 24

    inp_cells, reward_cell, punish_cell, reward_input_syns, punish_input_syns = \
        make_hh_network(input_size=agent.input_size)

    agent.init(init_v=-70, warmup=20, dt=1, input_cells=inp_cells, output_cells=inp_cells,
               reward_cell=reward_cell, punish_cell=punish_cell)
    print("Input neurons:", agent.input_cell_num)

    #syn_heatmap0 = HeatmapGraph(name="STILL", elements=inp_cells[0].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)
    #syn_heatmap1 = HeatmapGraph(name="UP", elements=inp_cells[1].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)
    #syn_heatmap2 = HeatmapGraph(name="DOWN", elements=inp_cells[2].syns, show_values=False, extract_func=lambda syn: syn.netcons[0].get_weight(), shape=input_shape, round_vals=6)

    move = -1
    last_agent_steptime = 0
    while True:
        env_step += 1
        #imshow(obs)
        #plt.pause(1e-3)

        env_step += 1
        obs, reward, done = step(obs, move, env_step)

        if reward != 0:
            agent.reward_step(reward=reward)

        # Make observation
        outputs = agent.step(observation=obs, output_type="rate", poisson=False)

        # moves from: -1 to 1
        move_vals = [o.value for o in outputs]
        move = np.argmax([o.value for o in outputs])

        if done == 1:
            obs, index = reset()
            env_step = 0
            print(env_step, 'reward:', reward, 'avg_reward:', round(np.average(rewards), 2),
                'move_vals', move_vals)
