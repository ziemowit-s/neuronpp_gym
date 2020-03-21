import math
import time
import matplotlib.pyplot as plt
import numpy as np
import gym 
from neuronpp.utils.network_status_graph import NetworkStatusGraph
from matplotlib.animation import FuncAnimation
from agents.cart_agent import CartAgent

plt.close('all')
env  =  gym.make('CartPole-v0')
n_episodes  =  100
env_vis  =  []

def env_render(env_vis):
    plt.figure()    
    plot  =  plt.imshow(env_vis[0])    
    plt.axis('off')    
    def  animate(i):    
        plot.set_data(env_vis[i])
    anim  =  FuncAnimation(plt.gcf(), animate, frames=len(env_vis), interval=100,
                           repeat=True, repeat_delay=20)
    # display(display_animation(anim, default_mode='loop')

#%%

AGENT_STEPSIZE = 50
LABELS = 2

agent = CartAgent(input_cell_num=4, input_size=4,
                  output_size=LABELS, max_hz=200, default_stepsize=AGENT_STEPSIZE)

agent.init(init_v=-70, warmup=10, dt=0.25)

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
action = 0
for i_episode in range(n_episodes):
    observation=env.reset()
    for t in range(100):
        env_vis.append(env.render(mode='rgb_array'))
        # print(observation)
        obs,reward,done,info=env.step(action)
        # print(obs)
        obs = np.array([1,1,1,1])*100
        output_spikes_ms=agent.step(observation=abs(obs.reshape((2,2))), reward=reward)
        print(output_spikes_ms)
        if t>10: reward=1
        else: reward=-1

        # plt.pause(1e-9)
        # agent.rec_input.plot(animate=True, position=(2, 2))
        graph.update_spikes(agent.sim.t)
        graph.update_weights('w')
        if done:
            print("Episode  finished  at  t{}".format(t+1))
            break

