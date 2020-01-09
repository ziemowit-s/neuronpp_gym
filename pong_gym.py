import cv2
import gym
import time
import numpy as np
from neuron import h
from neuron.units import mV
from neuronpp.utils.run_sim import RunSim
import matplotlib.pyplot as plt

from agents.neuromod_pong_agent import NeuromodPongAgent


# ACTIONS:
# 0 NOOP - no action
# 1 FIRE - no action
# 2 RIGHT - go up
# 3 LEFT - go down
# 4 RIGHTFIRE - go up
# 5 LEFTFIRE - go down

# OBSERVATION
# data for 3rd (last) array: obs[:,:,2]
# all colors: 0, 17, 74, 92, 236
# set to 0/255 only 2 colors

# REWARD
# lost=-1
# game=0
# win=1

# DONE
# done TRUE if >20 points


def show(obs):
    cv2.imshow("abc", obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise GeneratorExit("OpenCV image show stopped.")
    plt.imshow(obs, vmin=0, vmax=1)
    plt.show()


def prep_obs(obs, pivot=17):
    obs = obs[:, :, 2]  # take only one RGB array
    obs = obs[34:193, :]  # remove boundaries and score
    obs = cv2.resize(obs, (50,50))
    obs[obs <= pivot] = 0
    obs[obs > pivot] = 1
    obs = np.reshape(obs, [obs.shape[0] * obs.shape[1]])
    return obs


with gym.make('Pong-v0') as env:
    env.reset()
    #agent = NeuromodPongAgent(input_size=100, pyr_synapse=100, inh_synapse=100)

    # Warmup env
    for i in range(21):
        env.step(0)

    # Play
    action = 0
    for i in range(1000):
        env.render(mode='human')
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = prep_obs(obs, pivot=17)  # 2500 pixels
        print(obs.size)

        if done or reward > 0:
            print('DONE:', reward, done)
            env.reset()

        #action = agent.step(observation=obs[:100], reward=reward)
        time.sleep(0.01)

    #agent.retina_rec.plot()
    plt.show()
