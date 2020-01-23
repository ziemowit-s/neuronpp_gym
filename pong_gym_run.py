import time
import numpy as np
import matplotlib.pyplot as plt
from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.3

AGENT_STEPSIZE = 5
ENV_STEPSIZE = 5

ENV_STEPSIZE = ENV_STEPSIZE/1000


def get_action(current_time, up_moves, down_moves):
    action = 0
    #print('rel time:', current_time_relative)

    if len(up_moves) > 0 and current_time >= up_moves[0]:
        action = 2
        up_moves = up_moves[1:]

    if len(down_moves) > 0:
        if current_time >= down_moves[0]:
            if action == 2:
                action = 99
            else:
                action = 3
            down_moves = down_moves[1:]

    if action != 0:
        a = ""
        if action == 2:
            a = "UP"
        if action == 3:
            a = "DOWN"
        if action == 99:
            action = np.random.randint(2, 4)
            a = "RAND"
        print('Time:', round(agent.sim.t), "ms", "Move:", a)

    return action, up_moves, down_moves


if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_cell_num=6, input_size=input_size, output_size=2, max_hz=300, stepsize=AGENT_STEPSIZE, warmup=200,
                       weight=0.01, motor_weight=0.01)
    print('input_size', input_size)

    agent_compute_time = 0
    agent_observe = True
    up_moves = []
    down_moves = []
    action = 0
    start_time = time.time()
    while True:

        env.render()
        obs, reward, done, info = env.step(action)
        action = 0
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        if reward != 0:
            print('reward:', reward)
            if reward < 0:
                reset(env, SCREEN_RATIO)

        if done:
            reset(env, SCREEN_RATIO)

        # Make moves
        t = time.time()
        current_time = (t - start_time)
        current_time_relative = (t - agent_compute_time)
        print(current_time_relative)
        action, up_moves, down_moves = get_action(current_time_relative, up_moves, down_moves)

        # Sent observation to the Agent every AGENT_STEPSIZE ms
        if agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            moves = agent.step(observation=obs, reward=reward)
            up_moves.extend([m/1000 for m in moves[0]])
            down_moves.extend([m/1000 for m in moves[1]])

            if len(up_moves) > 0 or len(down_moves) > 0:
                print("moves:", up_moves, down_moves)

            agent_compute_time = time.time()

        #agent.rec.plot()
        #plt.pause(0.5)
        #plt.close()
    env.close()
