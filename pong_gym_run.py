import numpy as np
import time

from neuronpp.utils.network_status_graph import NetworkStatusGraph
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph

from agents.agent import ConvParam
from agents.sigma3_olfactory_agent import Sigma3OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.15
ENV_STEPSIZE = 10
AGENT_STEPSIZE = 30
RESET_AFTER = 24

ENV_STEPSIZE = ENV_STEPSIZE / 1000


def make_action(move):
    if move == 0:  # UP
        return 2
    elif move == 1:  # DOWN
        return 3
    else:  # NONE
        return 0


if __name__ == '__main__':
    env, input_shape = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = Sigma3OlfactoryAgent(output_cell_num=2, input_max_hz=20, stepsize=AGENT_STEPSIZE)
    agent.build(input_shape=input_shape,
                x_param=ConvParam(kernel_size=6, padding=0, stride=6),
                y_param=ConvParam(kernel_size=6, padding=0, stride=6))
    agent.init(init_v=-70, warmup=10, dt=0.3)
    print("Input neurons:", agent.input_cell_num)

    # Create heatmap graph for input cells
    hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(4, 3))
    # Create network graph
    network_graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
    network_graph.plot()

    agent_compute_time = 0
    agent_observe = True
    start_time = time.time()

    move = -1
    reset_time = time.time()
    gain = 0
    while True:
        env.render()

        action = make_action(move)
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        time_from_reset = time.time() - reset_time
        if reward != 0 or done or time_from_reset > RESET_AFTER:

            # print('reward:', reward, 'time_from_reset:', time_from_reset)
            reset(env, SCREEN_RATIO)
            reset_time = time.time()
            if reward > 0 or time_from_reset > 20:
                gain += 1
                reward = 1
            elif reward < 0:
                # print("score:", gain)
                gain = 0

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)
        if agent_compute_time > 0:
            stepsize = current_time_relative * 1000
        else:
            stepsize = None

        # Sent observation to the Agent every ENV_STEPSIZE ms
        if reward != 0 or agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            # agent step
            outputs = agent.step(observation=obs, reward=reward, output_type="time", sort_func=lambda x: -x[1])
            move = -1
            if np.abs(outputs[0].value - outputs[1].value) > agent.sim.dt and outputs[0].value > -1:
                print(outputs)
                move = outputs[0].index

            # Update graphs
            network_graph.update_spikes(agent.sim.t)
            network_graph.update_weights('w')
            hitmap_graph.plot()

            # write time after agent step
            agent_compute_time = time.time()

        # make visuatization of mV on each cells by layers
        agent.rec_input.plot(animate=True, position=(4, 3))
        # agent.rec_output.plot(animate=True)
