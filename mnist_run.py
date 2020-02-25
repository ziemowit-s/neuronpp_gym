import time
import numpy as np
from neuronpp.utils.utils import show_connectivity_graph
import tensorflow as tf
import pylab as py
from agents.olfactory_agent import OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset
from scipy.ndimage import gaussian_filter

SCREEN_RATIO = 0.2  # 5 ms with: screen_ratio=0.2, step_size=3; 10sec env -> 0.3 sec network
AGENT_STEPSIZE = 10
# ENV_STEPSIZE = 100
THR = 100

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train, y_train = zip(*[(xs, ys) for xs,ys in zip(x_train, y_train) if ys in [0,1,2]])
index_list = []
for i,labels in enumerate(y_train): 
    if labels in [0,1,2]: 
        index_list.append(i)
x_train, y_train = x_train[index_list], y_train[index_list]
obj = py.imshow(x_train[0])

agent = OlfactoryAgent(input_cell_num=32, input_size=196, output_size=3, 
                       max_hz=300, default_stepsize=AGENT_STEPSIZE,
                       warmup=200, random_weight=True)

show_connectivity_graph(cells=agent.get_cells())
#%%
agent_compute_time = 0
agent_observe = True
start_time = time.time()

moves = []
reset_time = time.time()
gain = 0
index = 0
while True:
    if len(moves) > 0:
        action = moves[0][0]
        moves = moves[1:]
    else:
        action = 0
    y = y_train[index]
    # result = gaussian_filter(x_train[index], sigma=4)
    obs = x_train[index, ::2,::2]
    
    # write time before agent step
    current_time_relative = (time.time() - agent_compute_time)
    if agent_compute_time > 0:
        stepsize = current_time_relative * 1000
    else:
        stepsize = None
        
    # Sent observation to the Agent every ENV_STEPSIZE ms
    # agent step
    output = agent.step(observation=obs)
    
    spikes_list = np.zeros(3)
    for i,spikes in enumerate(output): 
        try: 
            spikes_list[i] = spikes[0]
        except: 
            spikes_list[i] = 1000
    if np.argmin(spikes_list) == y and np.min(spikes_list)!=1000:
        reward = 1
        print("i:", index,"reward recognized", y)
    else: 
        reward = -1
    agent.make_reward(reward=reward)
    # write time after agent step
    agent_compute_time = time.time()
    obj.set_data(obs)
    py.title('predict: '+ str(np.argmin(spikes_list)) + ' True: ' + str(y))
    py.draw()
    index+=1

    # plot output neurons
    #agent.rec_in.plot(animate=True, position=(4, 4))
    # plot input neurons
    # agent.rec_out.plot(position=(2, 2))
