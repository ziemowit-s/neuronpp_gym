import time
import numpy as np
from neuron import h

import matplotlib.pyplot as plt

from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation


ENV_STEP = 20000
SCREEN_RATIO = 0.1


def plot_spikes(v, agent, title=None, ax=None):
    if title:
        ax.set_title(title)

    ax.plot(agent.time_vec, v)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("voltage")
    plt.pause(0.0001)


if __name__ == '__main__':
    fig, axs = plt.subplots(1, clear=True)
    fig.show()
    plt.show()

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_size=input_size, max_hz=300, stepsize=100, finalize_step=5, warmup=200)
    print('input_size', input_size)

    action = 0
    for i in range(ENV_STEP):

        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=False)
        if done or reward != 0:
            print('reward:', reward, "done:", done)
            env.reset()

        output = agent.step(observation=obs, reward=reward)
        print("output max:", np.max(output))
        plot_spikes(output, agent=agent, ax=axs)

    agent.output_rec.plot()
    plt.show()
    env.close()
