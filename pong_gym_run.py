import numpy as np
import time

from neuronpp.utils.network_status_graph import NetworkStatusGraph
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph

from agents.agent import Kernel
from agents.sigma3_olfactory_agent import Sigma3OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.15
ENV_STEPSIZE = 10  # in ms
AGENT_STEPSIZE = 30  # in ms

RESET_AFTER = 24  # in seconds
REWARD_IF_PASS = 20  # in seconds


def make_action(move):
    if move == 0:  # UP
        return 2
    elif move == 1:  # DOWN
        return 3
    else:  # NONE
        return 0


if __name__ == '__main__':
    ENV_STEPSIZE = ENV_STEPSIZE / 1000

    env, input_shape = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = Sigma3OlfactoryAgent(output_cell_num=2, input_max_hz=20, default_stepsize=AGENT_STEPSIZE)
    agent.build(input_shape=input_shape,
                x_kernel=Kernel(size=6, padding=0, stride=6),
                y_kernel=Kernel(size=6, padding=0, stride=6))
    agent.init(init_v=-70, warmup=10, dt=0.3)
    print("Input neurons:", agent.input_cell_num)

    # Create heatmap graph for input cells
    hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(4, 3))
    # Create network graph
    network_graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
    network_graph.plot()

    move = -1
    last_agent_steptime = 0
    reset_time = time.time()
    while True:
        env.render()

        action = make_action(move)
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        # Make reward if play at leas REWARD_OF_PASS time
        time_from_reset = time.time() - reset_time
        if reward != 0 or done or time_from_reset > RESET_AFTER:
            reset(env, SCREEN_RATIO)
            reset_time = time.time()

            if time_from_reset > REWARD_IF_PASS:
                reward = 1


        # Make observation
        curr_relative_time = time.time() - last_agent_steptime
        if last_agent_steptime == 0 or curr_relative_time > ENV_STEPSIZE:
            outputs = agent.step(observation=obs, output_type="time", sort_func=lambda x: -x[1])

            output_time_diff = np.abs(outputs[0].value - outputs[1].value)

            move = -1
            if output_time_diff > agent.sim.dt and outputs[0].value > -1:
                print(outputs)
                move = outputs[0].index
            last_agent_steptime = time.time()

        # Make reward
        if reward != 0:
            agent.reward_step(reward=reward, stepsize=50)

        # Update graphs
        network_graph.update_spikes(agent.sim.t)
        network_graph.update_weights('w')
        hitmap_graph.plot()

        # write time after agent step
        last_agent_steptime = time.time()

        # make visuatization of mV on each cells by layers
        agent.rec_input.plot(animate=True, position=(4, 3))
        # agent.rec_output.plot(animate=True)
