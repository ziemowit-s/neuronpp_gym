import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt

# PONG PARAMS
# ACTIONS:
# 0 NOOP - no action
# 1 FIRE - no action
# 2 RIGHT - go up
# 3 LEFT - go down
# 4 RIGHTFIRE - go up
# 5 LEFTFIRE - go down

# OBSERVATION
# data for 3rd (last) array: obs[:,:,2]
# all colors: 0, 17, 74, 92, 236
# set to 0/255 only 2 colors

# REWARD
# lost=-1
# game=0
# win=1

# DONE
# done TRUE if >20 points


def show(obs):
    cv2.imshow("abc", obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise GeneratorExit("OpenCV image show stopped.")
    plt.imshow(obs, vmin=0, vmax=1)
    plt.show()


def prep_obs(obs, pivot=17, shape=(45, 45), show=False):
    obs = obs[:, :, 2]  # take only one RGB array
    obs = obs[34:193, :]  # remove boundaries and score
    obs = cv2.resize(obs, shape)
    obs[obs <= pivot] = 0
    obs[obs > pivot] = 1

    if show:
        plt.imshow(obs, vmin=0, vmax=1)
        plt.show()

    obs = np.reshape(obs, [obs.shape[0] * obs.shape[1]])
    return obs


def get_env(name):
    env = gym.make(name)
    env.reset()

    # Warmup evironment (required for 2 paddles to appear on the screen)
    for _ in range(21):
        obs, *_ = env.step(0)
    obs = prep_obs(obs)

    return env, obs.size
