import cv2
import gym
import numpy as np

from neuronpp.cells.cell import Cell
from neuronpp.core.distributions import UniformDist
from neuronpp.utils.iclamp import IClamp
from neuronpp.utils.record import Record

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


def get_cell_with_iclamps_as_synapses(warmup, dur, name, input_size, low=0.08, high=0.1):
    # Create cell
    cell = Cell(name=name)
    soma = cell.add_sec("soma", diam=10, l=10, nseg=5)
    dend = cell.add_sec("dend", diam=5, l=100, nseg=50)
    cell.connect_secs(child=dend, parent=soma)
    cell.insert("hh")

    # Make synapses
    iclamp_syns = []
    for i in range(input_size):
        iclamp = IClamp(segment=dend(np.random.rand()))
        syn = iclamp.stim(delay=warmup, dur=dur, amp=UniformDist(low=low, high=high))
        iclamp_syns.append(syn)

    # Make recording
    rec = Record(elements=soma(0.5), variables='v')
    cell.make_spike_detector(seg=soma(0.5))
    return cell, iclamp_syns, rec


def get_env_action_number(move):
    if move == 0:  # UP
        return 2
    elif move == 1:  # DOWN
        return 3
    else:  # NONE
        return 0


def show_cv2(obs):
    if "int" in str(obs.dtype):
        obs = np.array(obs, dtype=float)
    cv2.imshow("abc", obs)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise GeneratorExit("OpenCV image show stopped.")


def prepare_pong_observation(obs, ratio, show=False, flatten=False):
    """
    :param obs:
    :param ratio:
    :param show:
        it True - show what agent is see (white pixels=1, black pixels=0)
    :return:
    """
    def make(obs_arr, obj):
        max_x = obs_arr.shape[0] - 1
        max_y = obs_arr.shape[1] - 1
        for x, y in obj:
            if x > max_x:
                x = max_x
            if y > max_y:
                y = max_y
            obs_arr[x, y] = 1

    obs = obs[:, :, 2]  # take only one RGB array
    obs = obs[34:193, 19:141]  # remove boundaries and score

    right_paddle = np.argwhere(obs == 92)
    left_paddle = np.argwhere(obs == 74)
    ball = np.argwhere(obs == 236)

    # left only the smallest possible middle parts of objects
    if right_paddle.size > 0:
        right_paddle = right_paddle[3:-3]
        if right_paddle.size > 0:
            right_paddle = np.unique(np.array(np.round(right_paddle*ratio), dtype=int), axis=0)
    if left_paddle.size > 0:
        left_paddle = left_paddle[3:-3]
        if left_paddle.size > 0:
            left_paddle = np.unique(np.array(np.round(left_paddle*ratio), dtype=int), axis=0)
    if ball.size > 0:
        max_col_index = np.max(ball, axis=0)[1]
        ball = np.array([b for b in ball if b[1] == max_col_index])
        if ball.size > 0:
            ball = np.unique(np.array(np.round(ball*ratio), dtype=int), axis=0)

    shape = np.array(np.round(np.array(obs.shape) * ratio), dtype=int)
    obs = np.zeros(shape, dtype=int)

    if right_paddle.size > 0:
        make(obs, right_paddle)
    if left_paddle.size > 0:
        make(obs, left_paddle)
    if ball.size > 0:
        make(obs, ball)

    if show:
        show_cv2(obs)
        #plt.imshow(obs, vmin=0, vmax=1)
        #plt.show()

    #obs = np.reshape(obs, [obs.shape[0] * obs.shape[1]])  # flatten
    if flatten:
        obs = obs.flatten()
    return obs


def reset(env, ratio, prepare_obs=True):
    env.reset()
    # Warmup evironment (required for 2 paddles to appear on the screen)
    obs = None
    for _ in range(21):
        obs, *_ = env.step(0)
    if prepare_obs:
        obs = prepare_pong_observation(obs, ratio)

    return obs


def get_env(name, ratio):
    env = gym.make(name)
    return env, reset(env, ratio)



