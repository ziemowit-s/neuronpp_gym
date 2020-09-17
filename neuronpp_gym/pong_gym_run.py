import time

from neuronpp_gym.agents.agent2d import Agent2D
from neuronpp_gym.agents.utils import make_ebner_network
from neuronpp_gym.core.agent_core import Kernel
from neuronpp_gym.utils import get_env, prepare_pong_observation, reset, get_env_action_number
from neuronpp.utils.graphs.spikes_heatmap_graph import SpikesHeatmapGraph

SCREEN_RATIO = 0.15
AGENT_STEP_AFTER = 10  # env steps
AGENT_STEPSIZE = 20  # in ms

RESET_AFTER = 100  # env steps
REWARD_IF_PASS = 80  # env steps


if __name__ == '__main__':
    env, obs = get_env('Pong-v0', ratio=SCREEN_RATIO)
    input_shape = obs.shape

    agent = Agent2D(input_max_hz=100, default_stepsize=AGENT_STEPSIZE)
    agent.build(input_shape=input_shape,
                x_kernel=Kernel(size=3, padding=0, stride=3),  # 18
                y_kernel=Kernel(size=4, padding=0, stride=4))  # 24

    inp_pop, out_pop, reward_syns, punish_syns = make_ebner_network(input_size=agent.input_size,
                                                                    input_cell_num=agent.input_cell_num)

    agent.init(init_v=-70, warmup=20, dt=0.2, input_cells=inp_pop.cells, output_cells=out_pop.cells,
               reward_syns=reward_syns, punish_syns=punish_syns)
    print("Input neurons:", agent.input_cell_num)

    # Create heatmap graph for input cells
    hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(6, 6))

    move = -1
    last_agent_steptime = 0
    env_step = 0
    while True:
        env.render()

        action = get_env_action_number(move)
        env_step += 1
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        if reward != 0 or done or env_step >= RESET_AFTER:
            if env_step >= REWARD_IF_PASS:
                reward = 1
            print('env step:', env_step)

            reset(env, SCREEN_RATIO)
            env_step = 0

        # Make observation
        move = -1
        if env_step % AGENT_STEP_AFTER == 0:
            outputs = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value,
                                 poisson=False)

            # Update graphs
            hitmap_graph.plot()

            # Update output text
            if (outputs[0].value - outputs[1].value) >= 1:
                move = outputs[0].index
                print("answer: %s" % move)

            # write time after agent step
            last_agent_steptime = time.time()

        if reward != 0:
            print('reward:', reward, 'time:', round(agent.sim.t))
            agent.reward_step(reward=reward, stepsize=10)

        # make visuatization of mV on each cells by layers
        # agent.rec_input.plot(animate=True, position=(6, 6))
        # agent.rec_output.plot(animate=True)
