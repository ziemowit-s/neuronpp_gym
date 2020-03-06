import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.utils.network_status_graph import NetworkStatusGraph

from agents.inhib_agent import InhibAgent


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
MNIST_LABELS = 3
SKIP_PIXELS = 2

x_train, y_train = mnist_prepare(num=MNIST_LABELS)
fig, ax = plt.subplots(1, 1)
obj = ax.imshow(x_train[0])

agent = InhibAgent(input_cell_num=16, input_size=x_train[:, ::SKIP_PIXELS].shape[1] ** 2,
                   output_size=MNIST_LABELS, max_hz=300, default_stepsize=AGENT_STEPSIZE)
agent.init(init_v=-70, warmup=10, dt=0.1)

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
    obs = x_train[index, ::SKIP_PIXELS, ::SKIP_PIXELS]

    # write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    output = agent.step(observation=obs, reward=reward, output_type="time", sort_func=lambda x: x)[0]
    predicted = -1
    if output.value > -1:
        predicted = output.index

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

    graph.update_spikes(agent.sim.t)
    graph.update_weights('w')

    #agent.rec_input.plot(animate=True, position=(4, 4))
    #agent.rec_out.plot(animate=True)
