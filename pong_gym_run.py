import time
import numpy as np

import matplotlib.pyplot as plt
from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.1

AGENT_STEPSIZE = 50


if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_size=input_size, max_hz=300, stepsize=AGENT_STEPSIZE, warmup=200)
    print('input_size', input_size)

    move_time = 0
    agent_observe = True
    up_moves = np.array([])
    down_moves = np.array([])
    action = 0
    while True:

        env.render()
        obs, reward, done, info = env.step(action)
        action = 0
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=False)

        if done or reward != 0:
            print('reward:', reward, "done:", done)

        if done:
            reset(env, SCREEN_RATIO)

        curr_time = time.time()*100
        if up_moves.size > 0:
            if (curr_time - move_time) >= up_moves[0]:
                action = 2
                up_moves = up_moves[1:]

        if down_moves.size > 0:
            if (curr_time - move_time) >= down_moves[0]:
                if action == 2:
                    action = np.random.randint(2,4)
                else:
                    action = 3
                down_moves = down_moves[1:]

        if ((curr_time - move_time)) > AGENT_STEPSIZE:
            moves = agent.step(observation=obs, reward=reward)
            up_moves = moves[0]
            down_moves = moves[1]
            move_time = time.time()*100
            print('up_move:', up_moves, 'down_move:', down_moves)
        time.sleep(0.05)

        #agent.rec.plot()
        #plt.pause(2)
        #plt.close()

    env.close()
