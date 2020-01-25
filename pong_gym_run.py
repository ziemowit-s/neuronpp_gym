import time
import numpy as np
import matplotlib.pyplot as plt
from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.2

AGENT_STEPSIZE = 50
ENV_STEPSIZE = 3
ENV_STEPSIZE = ENV_STEPSIZE/1000


if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_cell_num=6, input_size=input_size, output_size=2, max_hz=500, stepsize=AGENT_STEPSIZE, warmup=200,
                       weight=0.1, motor_weight=0.1, delay=2)
    print('input_size', input_size)

    agent_compute_time = 0
    agent_observe = True
    start_time = time.time()
    moves = []
    while True:

        env.render()
        if len(moves) > 0:
            action = moves[0][0]
            moves = moves[1:]
        else:
            action = 0
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        if reward != 0:
            print('reward:', reward)
        if done:
            reset(env, SCREEN_RATIO)

        current_time_relative = (time.time() - agent_compute_time)
        # Sent observation to the Agent every AGENT_STEPSIZE ms
        if agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            new_moves = agent.step(observation=obs, reward=reward)
            up_moves = new_moves[0]
            down_moves = new_moves[1]

            if len(up_moves) > 0 or len(down_moves) > 0:
                print("up:", up_moves, "down:", down_moves)

            new_moves = sorted([(2, m) for m in up_moves] + [(3, m) for m in down_moves], key=lambda x: x[1])
            moves.extend(new_moves)
            agent_compute_time = time.time()

        #agent.rec_out.plot()
        agent.rec_in.plot(max_plot_on_fig=10)
        plt.pause(0.5)
        plt.close()
    env.close()
