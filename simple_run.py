import queue
import time

import matplotlib.pyplot as plt
import numpy as np
from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from neuronpp.core.distributions import NormalTruncatedDist
from neuronpp.core.populations.population import Population
from neuronpp.utils.record import Record

from agents.agent1d import Agent1D


def get_img_plot():
    fig, ax = plt.subplots(1, 1)
    data = np.ones([2, 1], dtype=float)
    obj = ax.imshow(data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    return obj, ax


AGENT_STEPSIZE = 60  # in ms - how long agent will look on a single observation
EPSILON_OUTPUT = 1  # Min epsilon difference between outputs to decide that agent made decision
MAX_AVG_SIZE = 50  # Max size of average window to count accuracy


def input_cell_template():
    cell = Cell("cell")
    cell.add_sec("soma", diam=20, l=20, nseg=10)
    cell.add_sec("apic", diam=2, l=50, nseg=100)
    cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
    cell.insert('pas')
    cell.insert('hh')
    return cell


def output_cell_template():
    cell = Ebner2019AChDACell("ebner", compile_paths="agents/commons/mods/ebner2019 "
                                                     "agents/commons/mods/4p_ach_da_syns")

    cell.add_sec("soma", diam=20, l=20, nseg=10)
    cell.add_sec("apic", diam=2, l=50, nseg=100)
    cell.connect_secs(child="apic", parent="soma", child_loc=0, parent_loc=1)
    cell.make_default_mechanisms()
    return cell


# Create network
input_pop = Population("input")
input_pop.add_cells(num=2, cell_function=input_cell_template)

output_pop = Population("output")
output_pop.add_cells(num=2, cell_function=output_cell_template)

connector = output_pop.connect(rule="all", cell_connection_proba=0.5)
connector.set_source([c.filter_secs("soma")(0.5) for c in input_pop.cells])
connector.set_target([c.filter_secs("soma")(0.5) for c in output_pop.cells])

syn_adder = connector.add_synapse("Syn4PAChDa").add_netcon(weight=0.01)
syn_adder.add_point_process_params(w_pre_init=NormalTruncatedDist(mean=0.5, std=1 / 8),
                                   w_post_init=NormalTruncatedDist(mean=2, std=1 / 8),
                                   ACh_tau=1, Da_tau=1)

connector.add_synapse("SynACh").add_netcon(weight=1)
syn_adder = connector.add_synapse("SynDa").add_netcon(weight=1)
connector.group_synapses()


def set_pointer(syns):
    Ebner2019AChDACell.set_synaptic_pointers(syns[0], syns[1], syns[2])


connector.set_synaptic_function(func=set_pointer)
connector.build()

agent = Agent1D(input_max_hz=50, default_stepsize=AGENT_STEPSIZE)
agent.build(input_cells=input_pop.cells, output_cells=output_pop.cells)
agent.init(init_v=-70, warmup=100, dt=0.1)

# Show and update mnist image
imshow_obj, ax = get_img_plot()

# Run training
index = 0
reward = None
agent_compute_time = 0
avg_acc_fifo = queue.Queue(maxsize=MAX_AVG_SIZE)
while True:
    # get current obs and y
    obs = np.random.randint(low=0, high=2, size=2)
    if obs[0] == 1:
        obs[1] = 0
        y = 0
    else:
        obs[1] = 1
        y = 1

    # Write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    # Make step and get agent predictions
    predicted = -1
    outputs = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value,
                         poisson=True)

    if outputs[0].value != -1 and \
            (outputs[1].value == -1 or (outputs[0].value - outputs[1].value) >= EPSILON_OUTPUT):
        predicted = outputs[0].index

    # Clear average accuracy fifo
    if avg_acc_fifo.qsize() == MAX_AVG_SIZE:
        avg_acc_fifo.get()

    # Make reward
    if predicted == y:
        avg_acc_fifo.put(1)
        reward = 1
        print("answer:", predicted, "OK!")
    else:
        avg_acc_fifo.put(0)
        reward = -1
        print("answer:", predicted)

    # Update image
    obs = obs.reshape([2, 1])
    imshow_obj.set_data(obs)
    avg_accuracy = round(np.average(list(avg_acc_fifo.queue)), 2)
    ax.set_title('Predicted: %s True: %s, AVG_ACC: %s' % (predicted, y, avg_accuracy))

    # Make reward step
    agent.reward_step(reward=reward, stepsize=50)

    # Write time after agent step
    agent_compute_time = time.time()

    # increment mnist image index
    index += 1
