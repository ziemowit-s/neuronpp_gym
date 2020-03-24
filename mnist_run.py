import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.utils.network_status_graph import NetworkStatusGraph
from neuronpp.utils.spikes_heatmap_graph import SpikesHeatmapGraph

from agents.agent import ConvParam
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


def make_mnist_imshow(x_train, agent):
    """
    Prepare image show to update on each steps. It will also show receptive field of each input neuron
    :param agent:
    :param x_train:
        whole mnist trainset
    :return:
    """
    fig, ax = plt.subplots(1, 1)
    template = agent.pad_observation(x_train[0])

    x_ticks = np.arange(0, template.shape[0], agent.x_stride)
    y_ticks = np.arange(0, template.shape[1], agent.y_stride)

    ax.set_xticks(x_ticks)
    ax.set_xticks([i for i in range(template.shape[0])], minor=True)
    ax.set_yticks(y_ticks)
    ax.set_yticks([i for i in range(template.shape[1])], minor=True)

    obj = ax.imshow(x_train[0], cmap=plt.get_cmap('gray'), extent=[0, template.shape[0], 0, template.shape[1]])
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    return obj, ax


AGENT_STEPSIZE = 60  # in ms - how long agent will look on a single mnist image
MNIST_LABELS = 3  # how much mnist digits we want
SKIP_PIXELS = 2  # how many pixels on mnist we want to skip (image will make smaller)

# Prepare mnist dataset
x_train, y_train = mnist_prepare(num=MNIST_LABELS)
x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]

# Create Agent
agent = EbnerAgent(output_cell_num=MNIST_LABELS, input_max_hz=100, stepsize=AGENT_STEPSIZE)
agent.build(input_shape=x_train.shape[1:],
            x_param=ConvParam(kernel_size=4, padding=1, stride=4),
            y_param=ConvParam(kernel_size=4, padding=1, stride=4))
agent.init(init_v=-80, warmup=2000, dt=0.2)
print("Input neurons:", agent.input_cell_num)

# Show and update mnist image
imshow_obj, ax = make_mnist_imshow(x_train, agent)

# Create heatmap graph for input cells
hitmap_shape = int(np.ceil(np.sqrt(agent.input_cell_num)))
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

    # Update graphs
    network_graph.update_spikes(agent.sim.t)
    network_graph.update_weights('w')
    hitmap_graph.plot()

    # Update visualizations
    imshow_obj.set_data(agent.pad_observation(obs))
    ax.set_title('Predicted: %s True: %s' % (predicted, y))
    plt.draw()
    plt.pause(1e-9)

    # Make reward step
    agent.make_reward_step(reward=reward, stepsize=50)
    # Write time after agent step
    agent_compute_time = time.time()

    # increment mnist image index
    index += 1

    # make visuatization of mV on each cells by layers
    # agent.rec_input.plot(animate=True, position=(4, 4))
    # agent.rec_output.plot(animate=True)
