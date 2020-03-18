import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.utils.network_status_graph import NetworkStatusGraph
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph

from agents.ebner_agent import EbnerAgent
from agents.ebner_olfactory_agent import EbnerOlfactoryAgent


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


def make_mnist_imshow(x_train):
    """
    Prepare image show to update on each steps. It will also show receptive field of each input neuron
    :param x_train:
        whole mnist trainset
    :return:
    """
    fig, ax = plt.subplots(1, 1)

    x_ticks = np.arange(0, x_train.shape[1], x_pixel_size)
    y_ticks = np.arange(0, x_train.shape[2], y_pixel_size)
    ax.set_xticks(x_ticks)
    ax.set_xticks([i for i in range(x_train.shape[1])], minor=True)
    ax.set_yticks(y_ticks)
    ax.set_yticks([i for i in range(x_train.shape[2])], minor=True)

    obj = ax.imshow(x_train[0], cmap=plt.get_cmap('gray'), extent=[0, x_train.shape[1], 0, x_train.shape[2]])
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    return obj, ax


AGENT_STEPSIZE = 50  # in ms - how long agent will look on a single mnist image
MNIST_LABELS = 3  # how much mnist digits we want
SKIP_PIXELS = 2  # how many pixels on mnist we want to skip (image will make smaller)
INPUT_CELL_NUM = 36  # how much input cells agent will have

# Prepare mnist dataset
x_train, y_train = mnist_prepare(num=MNIST_LABELS)
x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]
input_size = x_train.shape[1] * x_train.shape[2]

# Create Agent
agent = EbnerAgent(input_cell_num=INPUT_CELL_NUM, input_size=input_size,
                   output_size=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
agent.init(init_v=-80, warmup=2000, dt=0.3)

# Print how many pixels for each input cell
x_pixel_size, y_pixel_size = agent.get_input_cell_observation_shape(x_train[0])
input_syn_per_cell = int(np.ceil(input_size / INPUT_CELL_NUM))
print('pixels per input cell:', input_syn_per_cell)

# Show and update mnist image
obj, ax = make_mnist_imshow(x_train)

# Create heatmap graph for input cells
hitmap_shape = int(np.ceil(np.sqrt(INPUT_CELL_NUM)))
hitmap_graph = SpikesHeatmapGraph(name="Input Cells", cells=agent.input_cells, shape=(hitmap_shape, hitmap_shape))

# Create network graph
network_graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
network_graph.plot()

# %%
index = 0
reward = None
agent_compute_time = 0

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
    outputs = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value)
    output = outputs[0]
    predicted = -1
    if output.value > -1:
        predicted = output.index

    # Make reward
    if predicted == y:
        reward = 1
        print("i:", index, "reward recognized", y)
    else:
        reward = -1
    agent.make_reward_step(reward=reward)

    # Write time after agent step
    agent_compute_time = time.time()

    # Update graphs
    network_graph.update_spikes(agent.sim.t)
    network_graph.update_weights('w')
    hitmap_graph.plot()

    # Update visualizations
    obj.set_data(obs)
    ax.set_title('Predicted: %s True: %s' % (predicted, y))
    plt.draw()
    plt.pause(1e-9)

    # increment mnist image index
    index += 1

    # make visuatization of mV on each cells by layers
    # agent.rec_input.plot(animate=True, position=(4, 4))
    # agent.rec_output.plot(animate=True)
