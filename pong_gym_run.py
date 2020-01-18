import time

import matplotlib.pyplot as plt

from agents.ebner_agent import EbnerAgent
from utils import get_env, prepare_pong_observation


ENV_STEP = 20000
RATIO = 0.1

if __name__ == '__main__':

    env, input_size = get_env('Pong-v0', ratio=RATIO)
    print('input_size', input_size)
    agent = EbnerAgent(input_size=input_size, max_hz=300, stepsize=100, finalize_step=5, warmup=200)

    action = 0
    for i in range(ENV_STEP):

        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = prepare_pong_observation(obs, ratio=RATIO, show=False)
        if done or reward != 0:
            print('reward:', reward, "done:", done)
            env.reset()

        time.sleep(0.01)
        action = agent.step(observation=obs, reward=reward)

    agent.output_rec.plot()
    plt.show()
    env.close()
