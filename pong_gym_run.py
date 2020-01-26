import time
from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.2
AGENT_STEPSIZE = 3
ENV_STEPSIZE = 3
ENV_STEPSIZE = ENV_STEPSIZE/1000


if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = EbnerAgent(input_cell_num=9, input_size=input_size, output_size=2, max_hz=300, stepsize=AGENT_STEPSIZE, warmup=200)
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
            reset(env, SCREEN_RATIO)
        if done:
            reset(env, SCREEN_RATIO)

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)

        # Sent observation to the Agent every ENV_STEPSIZE ms
        if agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            # agent step
            new_moves = agent.step(observation=obs, reward=reward)

            up_moves = new_moves[0]
            down_moves = new_moves[1]
            # print moves
            if len(up_moves) > 0 or len(down_moves) > 0:
                print("up:", up_moves, "down:", down_moves)

                # extend moves list with new moves
                new_moves = sorted([(2, m) for m in up_moves] + [(3, m) for m in down_moves], key=lambda x: x[1])
                moves.extend(new_moves)

            # write time after agent step
            agent_compute_time = time.time()

        # plot input neurons
        agent.rec_out.plot(position=(2, 2))
        # plot output neurons
        agent.rec_in.plot(position=(3, 3))
