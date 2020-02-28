import time
from neuronpp.utils.utils import key_release_listener

from agents.olfactory_agent import OlfactoryAgent
from utils import get_env, prepare_pong_observation, reset

SCREEN_RATIO = 0.2  # 5 ms with: screen_ratio=0.2, step_size=3; 10sec env -> 0.3 sec network
ENV_STEPSIZE = 3
AGENT_STEPSIZE = 3
RESET_AFTER = 16

ENV_STEPSIZE = ENV_STEPSIZE / 1000


key_pressed = ['']


def key_pressed_func(key):
    print(key)
    key_pressed[0] = key

key_release_listener(key_pressed_func)

if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=SCREEN_RATIO)
    agent = OlfactoryAgent(input_cell_num=9, input_size=input_size, output_size=2, max_hz=300, default_stepsize=AGENT_STEPSIZE, warmup=200)
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
            if reward < 0:
                print("score:", gain)
                gain = 0
            if reward > 0 or time_from_reset > 15:
                gain += 1

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)
        if agent_compute_time > 0:
            stepsize = current_time_relative * 1000
        else:
            stepsize = None

        # Reward for single move
        if key_pressed[0] == 'w':
            print('w')
            agent.make_reward(1)
            key_pressed[0] = ''

        # Sent observation to the Agent every ENV_STEPSIZE ms
        if reward != 0 or agent_compute_time == 0 or current_time_relative > ENV_STEPSIZE:
            # agent step
            new_moves = agent.step(observation=obs, reward=0)

            up_moves = new_moves[0]
            down_moves = new_moves[1]
            # print moves
            if len(up_moves) > 0 or len(down_moves) > 0:
                #print("up:", np.round(up_moves/1000, 4), "down:", np.round(down_moves/1000, 4))

                # extend moves list with new moves
                new_moves = sorted([(2, m) for m in up_moves] + [(3, m) for m in down_moves], key=lambda x: x[1])
                moves.extend(new_moves)

            # write time after agent step
            agent_compute_time = time.time()

        # plot output neurons
        #agent.rec_in.plot(animate=True, position=(4, 4))
        # plot input neurons
        # agent.rec_out.plot(position=(2, 2))
