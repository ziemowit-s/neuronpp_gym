import time
import numpy as np

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

AGENT_STEPSIZE = 10


if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_size=input_size, max_hz=300, stepsize=AGENT_STEPSIZE, warmup=200)
    print('input_size', input_size)

    move_time = 0
    agent_observe = True
    up_moves = np.array([])
    action = 0
    for i in range(ENV_STEP):

        env.render()
        obs, reward, done, info = env.step(action)
        action = 0
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=False)

        if done or reward != 0:
            print('reward:', reward, "done:", done)
            env.reset()

        curr_time = time.time()*100
        if up_moves.size > 0:
            if (curr_time - move_time) >= up_moves[0]:
                action = 2
                up_moves = up_moves[1:]

        if ((curr_time - move_time)) > AGENT_STEPSIZE:
            moves = agent.step(observation=obs, reward=reward)
            up_moves = moves[0]
            move_time = time.time()*100
        time.sleep(0.05)
        #agent.cell.plot_spikes()
        #plt.pause(1.0)
        #plt.close()
    env.close()
