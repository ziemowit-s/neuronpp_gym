import time
import numpy as np

from core.agent_core import Kernel
from agents.agent2d import Agent2D
from neuronpp.cells.ebner2019_cell import Ebner2019Cell
from utils import get_env, prepare_pong_observation, reset
from neuronpp.core.populations.population import Population
from neuronpp.core.distributions import UniformTruncatedDist
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell

SCREEN_RATIO = 0.15
AGENT_STEP_AFTER = 10  # env steps
AGENT_STEPSIZE = 20  # in ms

RESET_AFTER = 100  # env steps
REWARD_IF_PASS = 80  # env steps


def make_network(input_size, input_cell_num):
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
    con.add_synapse("Syn4PAChDa").add_netcon(weight=UniformTruncatedDist(low=0.01, high=0.1))
    con.add_synapse("SynACh").add_netcon(weight=0.1)
    con.add_synapse("SynDa").add_netcon(weight=0.1)
    con.set_synaptic_function(func=lambda syns: Ebner2019AChDACell.set_synaptic_pointers(*syns))
    con.group_synapses()
    con.build()
    return inp, out


def make_action(move):
    if move == 0:  # UP
        return 2
    elif move == 1:  # DOWN
        return 3
    else:  # NONE
        return 0


if __name__ == '__main__':
    env, input_shape = get_env('Pong-v0', ratio=SCREEN_RATIO)

    agent = Agent2D(input_max_hz=50, default_stepsize=AGENT_STEPSIZE)
    agent.build(input_shape=input_shape,
                x_kernel=Kernel(size=3, padding=0, stride=3),  # 18
                y_kernel=Kernel(size=4, padding=0, stride=4))  # 24

    inp_pop, out_pop = make_network(input_size=agent.input_size,
                                    input_cell_num=agent.input_cell_num)

    agent.init(init_v=-70, warmup=20, dt=0.2, input_cells=inp_pop.cells, output_cells=out_pop.cells)
    print("Input neurons:", agent.input_cell_num)

    # Create heatmap graph for input cells
    #hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(6, 6))

    move = -1
    last_agent_steptime = 0
    env_step = 0
    while True:
        env.render()

        action = make_action(move)
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
            #hitmap_graph.plot()

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
        #agent.rec_input.plot(animate=True, position=(6, 6))
        #agent.rec_output.plot(animate=True)
