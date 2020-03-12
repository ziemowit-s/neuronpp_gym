import numpy as np
import time

from agents.sigma3_olfactory_agent import Sigma3OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.3  # 5 ms with: screen_ratio=0.2, step_size=3; 10sec env -> 0.3 sec network
ENV_STEPSIZE = 10
AGENT_STEPSIZE = 3
RESET_AFTER = 24

ENV_STEPSIZE = ENV_STEPSIZE / 1000


def make_action(move):
    if move == 0:  # UP
        return 2
    elif move == 1:  # DOWN
        return 3
    else:  # NONE
        return 0


if __name__ == '__main__':
    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = Sigma3OlfactoryAgent(input_cell_num=9, input_size=input_size, output_size=2, input_max_hz=100,
                                 default_stepsize=AGENT_STEPSIZE)
    agent.init(init_v=-70, warmup=10, dt=0.1)
    print('input_size', input_size)

    agent_compute_time = 0
    agent_observe = True
    start_time = time.time()

    move = -1
    reset_time = time.time()
    gain = 0
    while True:
        env.render()

        action = make_action(move)
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        time_from_reset = time.time() - reset_time
        if reward != 0 or done or time_from_reset > RESET_AFTER:

            # print('reward:', reward, 'time_from_reset:', time_from_reset)
            reset(env, SCREEN_RATIO)
            reset_time = time.time()
            if reward > 0 or time_from_reset > 20:
                gain += 1
                reward = 1
            elif reward < 0:
                # print("score:", gain)
                gain = 0

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)
        if agent_compute_time > 0:
            stepsize = current_time_relative * 1000
        else:
            stepsize = None

        # Sent observation to the Agent every ENV_STEPSIZE ms
        if reward != 0 or agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            # agent step
            outputs = agent.step(observation=obs, reward=reward, output_type="time", sort_func=lambda x: -x[1])
            move = -1
            if np.abs(outputs[0].value - outputs[1].value) > agent.sim.dt and outputs[0].value > -1:
                print(outputs)
                move = outputs[0].index

            # write time after agent step
            agent_compute_time = time.time()

        # plot output neurons
        # agent.rec_in.plot(animate=True, position=(3, 3))
        # plot input neurons
        # agent.rec_out.plot(animate=True)
