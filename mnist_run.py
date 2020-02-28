import time
import pylab as plt
import numpy as np
import tensorflow as tf
from agents.olfactory_agent import OlfactoryAgent
from neuronpp.utils.utils import show_connectivity_graph

SCREEN_RATIO = 0.2  # 5 ms with: screen_ratio=0.2, step_size=3; 10sec env -> 0.3 sec network
AGENT_STEPSIZE = 10
# ENV_STEPSIZE = 100
THR = 100

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train, y_train = zip(*[(xs, ys) for xs,ys in zip(x_train, y_train) if ys in [0,1,2]])
index_list = []
for i, labels in enumerate(y_train):
    if labels in [0, 1, 2]:
        index_list.append(i)
x_train, y_train = x_train[index_list], y_train[index_list]
obj = plt.imshow(x_train[0])

agent = OlfactoryAgent(input_cell_num=9, input_size=196, output_size=3, max_hz=100, default_stepsize=AGENT_STEPSIZE, warmup=10)
#agent.show_connectivity_graph()
w_out = [pp.hoc.w for c in agent.output_cells for pp in c.pps]

# %%
agent_compute_time = 0
agent_observe = True
start_time = time.time()

reset_time = time.time()
gain = 0
index = 0
reward = None
while True:
    y = y_train[index]
    # result = gaussian_filter(x_train[index], sigma=4)
    obs = x_train[index, ::2, ::2]

    # write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None

    # Sent observation to the Agent every ENV_STEPSIZE ms
    # agent step
    output = agent.step(observation=obs, reward=reward)

    spikes_list = np.zeros(3)
    for i, spikes in enumerate(output):
        try:
            spikes_list[i] = spikes[0]
        except:
            spikes_list[i] = -1

    if np.argmin(spikes_list) == y and np.min(spikes_list) != -1:
        reward = 1
        print("i:", index, "reward recognized", y)
    else:
        reward = -1
    print('predict:', np.argmin(spikes_list), ' True:', y)

    # write time after agent step
    agent_compute_time = time.time()
    obj.set_data(obs)
    #plt.title('predict: ' + str(np.argmin(spikes_list)) + ' True: ' + str(y))
    plt.draw()
    plt.pause(1e-9)
    index += 1

    # plot output neurons
    #agent.rec_in.plot(animate=True, position=(4, 4))
    # plot input neurons
    #agent.rec_out.plot(animate=True)
