import queue
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.agent import Kernel
from agents.ebner_agent import EbnerAgent
from agents.sigma3_agent import Sigma3Agent
from neuronpp.utils.network_status_graph import NetworkStatusGraph
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph


def mnist_prepare(num=10):
    """
    :param num:
        number of mnist digits, default is 10 which is whole mnist
    :return:
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    index_list = []
    for i, labels in enumerate(y_train):
        if labels in list(np.arange(num)):
            index_list.append(i)

    x_train, y_train = x_train[index_list], y_train[index_list]
    return x_train, y_train


def make_mnist_imshow(x_train, agent):
    """
    Prepare image show to update on each steps. It will also show receptive field of each input neuron
    :param agent:
    :param x_train:
        whole mnist trainset
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    template = agent.pad_2d_observation(x_train[0])

    x_ticks = np.arange(0, template.shape[0], agent.x_kernel.stride)
    y_ticks = np.arange(0, template.shape[1], agent.y_kernel.stride)

    ax.set_xticks(x_ticks)
    ax.set_xticks([i for i in range(template.shape[0])], minor=True)
    ax.set_yticks(y_ticks)
    ax.set_yticks([i for i in range(template.shape[1])], minor=True)

    obj = ax.imshow(x_train[0], cmap=plt.get_cmap('gray'), extent=[0, template.shape[0], 0, template.shape[1]])
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    return obj, ax


AGENT_STEPSIZE = 60  # in ms - how long agent will look on a single mnist image
MNIST_LABELS = 2  # how much mnist digits we want
SKIP_PIXELS = 2  # how many pixels on mnist we want to skip (image will make smaller)

EPSILON_OUTPUT = 1  # Min epsilon difference between 2 the best output and the next one to decide if agent answered (otherwise answer: -1)
MAX_AVG_SIZE = 20  # Max size of average window to count accuracy

# Prepare mnist dataset
x_train, y_train = mnist_prepare(num=MNIST_LABELS)
x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]

# Create Agent
agent = EbnerAgent(output_cell_num=MNIST_LABELS, input_max_hz=80, default_stepsize=AGENT_STEPSIZE)
#agent = Sigma3Agent(output_cell_num=2, input_max_hz=80, default_stepsize=AGENT_STEPSIZE, netcon_weight=0.01, ach_tau=10, da_tau=20)
agent.build(input_shape=x_train.shape[1:],
            x_kernel=Kernel(size=2, padding=1, stride=2),
            y_kernel=Kernel(size=2, padding=1, stride=2))
#agent.init(init_v=-80, warmup=2000, dt=0.3)
agent.init(init_v=-70, warmup=100, dt=0.2)
print("Input neurons:", agent.input_cell_num)

# Show and update mnist image
imshow_obj, ax = make_mnist_imshow(x_train, agent)

# Create heatmap graph for input cells
#hitmap_shape = int(np.ceil(np.sqrt(agent.input_cell_num)))
#hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(hitmap_shape, hitmap_shape))

# Get output synapses
syns0 = [s for s in agent.output_cells[0].syns if "synach" not in s.name.lower() and "synda" not in s.name.lower()]
#syns1 = [s for s in agent.output_cells[1].syns if "synach" not in s.name.lower() and "synda" not in s.name.lower()]
#syns0 = [s for s in agent.output_cells[0].syns]
#syns1 = [s for s in agent.output_cells[1].syns]
cell0_weight_graph = WeightsHeatmapGraph(name="Cell 0 weights", syns=syns0, shape=(8, 8))
#cell1_weight_graph = WeightsHeatmapGraph(name="Cell 1 weights", syns=syns1, shape=(8, 8))

# %%
index = 0
reward = None
agent_compute_time = 0
accuracy_fifo = queue.Queue(maxsize=MAX_AVG_SIZE)

while True:
    # Get current mnist data
    obs = x_train[index]
    y = y_train[index]

    # Write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    # Make step and get agent predictions
    predicted = -1
    outputs = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value, poisson=True)

    # Update output text
    output_txt = 'i: %s output: %s' % (index, " / ".join(["%s:%s" % (o.index, o.value) for o in outputs]))

    # Update visualizations
    imshow_obj.set_data(agent.pad_2d_observation(obs))
    avg_accuracy = round(np.average(accuracy_fifo.queue), 2)
    ax.set_title('%s: Predicted: %s True: %s, AVG_ACC: %s' % (index, predicted, y, avg_accuracy))
    plt.draw()
    plt.pause(1e-9)

    # Update graphs
    if index % 10 == 0:
        cell0_weight_graph.plot()
        #cell1_weight_graph.plot()

    if (outputs[0].value - outputs[1].value) >= EPSILON_OUTPUT:
        predicted = outputs[0].index
        output_txt = "%s answer: %s" % (output_txt, predicted)

    print(output_txt)

    # Update accuracy
    if accuracy_fifo.qsize() == MAX_AVG_SIZE:
        accuracy_fifo.get()
    if predicted == y:
        accuracy_fifo.put(1)
    else:
        accuracy_fifo.put(0)

    # Make reward
    if predicted == -1:
        reward = np.random.randint(-1, 2, 1)[0]
        print("random rew:", reward)
    elif predicted == y:
        reward = 1
        print("CORRECT!")
    else:
        reward = -1

    agent.reward_step(reward=reward, stepsize=50)

    # Write time after agent step
    agent_compute_time = time.time()

    # increment mnist image index
    index += 1

    # make visuatization of mV on each cells by layers
    #agent.rec_input.plot(animate=True, position=(4, 4))
    #agent.rec_output.plot(animate=True)
