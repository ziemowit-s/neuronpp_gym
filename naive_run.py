import queue
import time
import numpy as np
import matplotlib.pyplot as plt

from agents.ebner_agent import EbnerAgent
from agents.sigma3_agent import Sigma3Agent
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph


def make_image():
    fig, ax = plt.subplots(1, 1)
    data = np.ones([2, 1], dtype=float)
    obj = ax.imshow(data, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    return obj, ax


AGENT_STEPSIZE = 60  # in ms - how long agent will look on a single mnist image
EPSILON_OUTPUT = 1  # Min epsilon difference between 2 the best output and the next one to decide if agent answered (otherwise answer: -1)
MAX_AVG_SIZE = 50  # Max size of average window to count accuracy

# Create Agent
agent = EbnerAgent(output_cell_num=2, input_max_hz=50, netcon_weight=0.01, default_stepsize=AGENT_STEPSIZE)
agent.build(input_shape=2)
agent.init(init_v=-70, warmup=100, dt=0.1)
print("Input neurons:", agent.input_cell_num)

# Show and update mnist image
imshow_obj, ax = make_image()

# Create heatmap graph for input cells
hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells)

index = 0
reward = None
agent_compute_time = 0
avg_acc_fifo = queue.Queue(maxsize=MAX_AVG_SIZE)
while True:
    # Get current mnist data
    obs = np.random.randint(low=0, high=2, size=2)
    if obs[0] == 1:
        obs[1] = 0
        y = 1
    else:
        obs[1] = 1
        y = 0

    # Write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    # Make step and get agent predictions
    predicted = -1
    outputs = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value, poisson=True)

    print('output:', " / ".join(["%s:%s" % (o.index, o.value) for o in outputs]))

    syns0 = [round(s.hoc.w, 2) for s in agent.output_cells[0].syns if
             "synach" not in s.name.lower() and "synda" not in s.name.lower()]
    syns1 = [round(s.hoc.w, 2) for s in agent.output_cells[1].syns if
             "synach" not in s.name.lower() and "synda" not in s.name.lower()]

    print("c1:", syns0, "c2:", syns1)

    if (outputs[0].value - outputs[1].value) >= EPSILON_OUTPUT:
        predicted = outputs[0].index
        print("answer:", predicted)

    # Make reward
    if avg_acc_fifo.qsize() == MAX_AVG_SIZE:
        avg_acc_fifo.get()
    if predicted == y:
        avg_acc_fifo.put(1)
        reward = 1
        print("i:", index, "value:", y)
    else:
        avg_acc_fifo.put(0)
        reward = -1

    # Update hitmap
    hitmap_graph.plot()

    # Update image
    obs = obs.reshape([2, 1])
    imshow_obj.set_data(obs)
    avg_accuracy = round(np.average(list(avg_acc_fifo.queue)), 2)
    ax.set_title('Predicted: %s True: %s, AVG_ACC: %s' % (predicted, y, avg_accuracy))
    plt.draw()
    plt.pause(1e-9)

    # Make reward step
    agent.reward_step(reward=reward, stepsize=50)

    # Write time after agent step
    agent_compute_time = time.time()

    # increment mnist image index
    index += 1

    # make visuatization of mV on each cells by layers
    # agent.rec_input.plot(animate=True)
    agent.rec_output.plot(animate=True)
