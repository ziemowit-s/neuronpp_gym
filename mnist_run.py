import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from agents.inhib_agent import InhibAgent
from neuronpp.utils.plot_network_status import PlotNetworkStatus


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


AGENT_STEPSIZE = 25
labels_from_mnist = 3
skip = 2

x_train, y_train = mnist_prepare(num=labels_from_mnist)
fig, ax = plt.subplots(1, 1)
obj = ax.imshow(x_train[0])

agent = InhibAgent(input_cell_num=12, input_size=x_train[:, ::skip].shape[1] ** 2,
                   output_size=labels_from_mnist, max_hz=1000,
                   default_stepsize=AGENT_STEPSIZE, warmup=10)

agent_compute_time = 0
agent_observe = True
start_time = time.time()
reset_time = time.time()
gain = 0
index = 0
reward = None

cells = agent.get_cells()
graph = PlotNetworkStatus(cells)

# %%
while True:
    y = y_train[index]
    obs = x_train[index, ::skip, ::skip]

    # write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    output_spikes_ms = agent.step(observation=obs, reward=reward)
    if len(output_spikes_ms) == 0:
        predicted = -1
    else:
        predicted = sorted([(i, ms[0] if len(ms) > 0 else math.inf) for i, ms in enumerate(output_spikes_ms)])[0]
        predicted = predicted[0] if predicted[1] != math.inf else -1

    if predicted == y:
        reward = 1
        print("i:", index, "reward recognized", y)
    else:
        reward = -1

    # write time after agent step
    agent_compute_time = time.time()

    # update image
    obj.set_data(obs)
    ax.set_title('Predicted: %s True: %s' % (predicted, y))
    plt.draw()
    plt.pause(1e-9)
    index += 1

    # plot output neurons
    # agent.rec_pattern.plot(animate=True, position=(2,1))
    # plot input neurons
    #agent.rec_out.plot(animate=True)
