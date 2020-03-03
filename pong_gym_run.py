import time

from agents.sigma3_olfactory_agent import Sigma3OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.3  # 5 ms with: screen_ratio=0.2, step_size=3; 10sec env -> 0.3 sec network
ENV_STEPSIZE = 10
AGENT_STEPSIZE = 3
RESET_AFTER = 24

ENV_STEPSIZE = ENV_STEPSIZE / 1000

if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = Sigma3OlfactoryAgent(input_cell_num=9, input_size=input_size, output_size=2, max_hz=100, default_stepsize=AGENT_STEPSIZE, warmup=10)
    agent.show_connectivity_graph()
    print('input_size', input_size)

    agent_compute_time = 0
    agent_observe = True
    start_time = time.time()

    moves = []
    reset_time = time.time()
    gain = 0
    while True:
        env.render()
        if len(moves) > 0:
            action = moves[0][0]
            moves = moves[1:]
        else:
            action = 0
        obs, reward, done, info = env.step(action)
        obs = prepare_pong_observation(obs, ratio=SCREEN_RATIO, show=True)

        time_from_reset = time.time() - reset_time
        if reward != 0 or done or time_from_reset > RESET_AFTER:
            print('reward:', reward, 'time_from_reset:', time_from_reset)
            reset(env, SCREEN_RATIO)
            reset_time = time.time()
            if reward > 0 or time_from_reset > 20:
                gain += 1
                reward = 1
            elif reward < 0:
                print("score:", gain)
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
            new_moves = agent.step(observation=obs, reward=reward)

            up_moves = new_moves[0]
            down_moves = new_moves[1]
            # print moves
            if len(up_moves) > 0 or len(down_moves) > 0:
                # extend moves list with new moves
                new_moves = sorted([(2, m) for m in up_moves] + [(3, m) for m in down_moves], key=lambda x: x[1])
                moves.extend(new_moves)

            # write time after agent step
            agent_compute_time = time.time()

        # plot output neurons
        #agent.rec_in.plot(animate=True, position=(3, 3))
        # plot input neurons
        #agent.rec_out.plot(animate=True)
