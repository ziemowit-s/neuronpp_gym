import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.utils.network_status_graph import NetworkStatusGraph

from agents.ebner_agent import EbnerAgent


def mnist_prepare(num=10):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    index_list = []
    for i, labels in enumerate(y_train):
        if labels in list(np.arange(num)):
            index_list.append(i)

    x_train, y_train = x_train[index_list], y_train[index_list]
    return x_train, y_train


def make_imshow(x_train):
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


AGENT_STEPSIZE = 100
MNIST_LABELS = 3
SKIP_PIXELS = 2
INPUT_CELL_NUM = 36

x_train, y_train = mnist_prepare(num=MNIST_LABELS)
x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]
input_size = x_train.shape[1] * x_train.shape[2]

agent = EbnerAgent(input_cell_num=INPUT_CELL_NUM, input_size=input_size,
                   output_size=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
agent.init(init_v=-80, warmup=2000, dt=0.3)

x_pixel_size, y_pixel_size = agent.get_input_cell_observation_shape(x_train[0])
input_syn_per_cell = int(np.ceil(input_size / INPUT_CELL_NUM))
print('pixels per input cell:', input_syn_per_cell)

obj, ax = make_imshow(x_train)

agent_compute_time = 0
agent_observe = True
start_time = time.time()
reset_time = time.time()
gain = 0
index = 0
reward = None

graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
graph.plot()

# %%
while True:
    y = y_train[index]
    obs = x_train[index]

    # write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    output = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value)[0]
    predicted = -1
    if output.value > -1:
        predicted = output.index

    if predicted == y:
        reward = 1
        print("i:", index, "reward recognized", y)
    else:
        reward = -1
    agent.make_reward_step(reward=reward)

    # write time after agent step
    agent_compute_time = time.time()

    # update image
    obj.set_data(obs)
    ax.set_title('Predicted: %s True: %s' % (predicted, y))

    graph.update_spikes(agent.sim.t)
    graph.update_weights('w')

    plt.draw()
    plt.pause(1e-9)
    index += 1

    #agent.rec_input.plot(animate=True, position=(4, 4))
    #agent.rec_output.plot(animate=True)
