import argparse
import copy
import itertools
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.cells.cell import Cell
from neuronpp.utils.record import MarkerParams

from agents.agent import Kernel
from agents.ebner_agent import EbnerAgent
from agents.ebner_olfactory_agent import EbnerOlfactoryAgent

np.set_printoptions(precision=2)

# print(sys.path)
# sys.path.extend('/Users/igor/git/neuronpp/neuronpp')
for e in sys.path:
    print(e)
# print(sys.path)
found = False
for e in sys.path:
    if 'NEURON-7.7/nrn/lib/python' in e:
        found = True
        break
if not found:
    print("NEURON-7.7 path not found in sys.path; exiting")
    sys.exit(0)

AGENT_STEPSIZE = 100
MNIST_LABELS = 3
SKIP_PIXELS = 2
INPUT_CELL_NUM = 9
DT = 0.2
RESTING_POTENTIAL = -70


class QueueNP:
    def __init__(self, class_no: int, length: int = 250, alpha: float = 0.9, gamma: float = 0.99):
        self._queue = np.zeros(shape=length, dtype=int)
        self._length = length
        self._pushed = 0
        self._qval = 0
        self._alpha = alpha
        self._gamma = gamma
        self._class_no = class_no
        self._tp = 0
        self._fp = 1
        self._tn = 2
        self._fn = 3
        self._tfpn = np.zeros((class_no, 4))
        self._mcc = np.zeros(class_no)
        self._f1 = np.zeros(class_no)

    def push(self, r: float, true: int, pred: int, qval: float):
        self._queue[:-1] = self._queue[1:]
        self._queue[-1] = r
        self._pushed = np.min((self._pushed + 1, self._length))
        # todo perhaps modify to compute the qval inside the queue (?)
        self._qval = qval
        for k in range(self._class_no):
            if k == true:
                if pred == k:
                    self._tfpn[k][self._tp] += 1
                else:
                    self._tfpn[k][self._fn] += 1
            else:
                if pred == k:
                    self._tfpn[k][self._fp] += 1
                else:
                    self._tfpn[k][self._tn] += 1
        for k in range(self._class_no):
            try:
                self._f1[k] = 2 * self._tfpn[k][self._tp] / (
                        2 * self._tfpn[k][self._tp] + self._tfpn[k][self._fp] + self._tfpn[k][self._fn])
            except RuntimeWarning:
                self._f1[k] = 0
            # todo add exception for zero-value denominator
            try:
                self._mcc[k] = (self._tfpn[k][self._tp] * self._tfpn[k][self._tn] -
                                self._tfpn[k][self._fp] * self._tfpn[k][self._fn]) \
                               / np.sqrt((self._tfpn[k][self._tp] + self._tfpn[k][self._fp]) *
                                         (self._tfpn[k][self._tp] + self._tfpn[k][self._fn]) *
                                         (self._tfpn[k][self._tn] + self._tfpn[k][self._fp]) *
                                         (self._tfpn[k][self._tn] + self._tfpn[k][self._fn]))
            except RuntimeWarning:
                self._mcc[k] = 0

    def sum(self):
        return np.sum(self._queue[-self.pushed:])

    def ratio(self):
        if self.pushed == 0:
            return 0.0
        else:
            return self.sum() / self.pushed

    @property
    def length(self):
        return self._length

    @property
    def pushed(self):
        return self._pushed

    @property
    def queue(self):
        return self._queue

    @property
    def qval(self):
        return self._qval

    @property
    def f1(self):
        return self._f1

    @property
    def mcc(self):
        return self._mcc


class EbOlA(EbnerOlfactoryAgent):
    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_pop, output_pop = super()._build_network(input_cell_num=input_cell_num, input_size=input_size,
                                                       output_cell_num=output_cell_num)
        # info add some inhibitory networks to the output layer
        self.motor_inhibit = []
        self._make_motor_inhibition_cells(population_position=4, source=output_pop, netcon_weight=0.1)
        return input_pop, output_pop

    def _make_motor_inhibition_cells(self, population_position, source, netcon_weight):
        for k, motor in enumerate(source):
            cell = Cell('minh', compile_paths="agents/commons/mods/sigma3syn")
            cell.name = "minh_%s[%s][%s]" % (population_position, cell.name, k)
            self.motor_inhibit.append(cell)
            soma = cell.add_sec("soma", diam=3, l=3, nseg=1)
            cell.insert("pas")
            cell.insert("hh")
            cell.add_synapse(source=motor.filter_secs("soma")(0.5), netcon_weight=netcon_weight,
                             seg=soma(0.5), mod_name="ExcSigma3Exp2Syn")
            for m, src in enumerate(source):
                if m == k: continue
                src.add_synapse(source=cell.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                                mod_name="Exp2Syn", e=-90)


class EbnerAgentInhb(EbnerAgent):
    def __init__(self, output_cell_num, input_max_hz, netcon_weight=0.01, default_stepsize=20, ach_tau=50, da_tau=50):
        super().__init__(output_cell_num, input_max_hz, netcon_weight=netcon_weight,
                         default_stepsize=default_stepsize, ach_tau=ach_tau, da_tau=da_tau)
        self.motor_inhibit = []

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_pop, output_pop = super()._build_network(input_cell_num=input_cell_num, input_size=input_size,
                                                       output_cell_num=output_cell_num)
        self._make_motor_inhibition_cells(population_position=4, source=output_pop, netcon_weight=0.1)
        return input_pop, output_pop

    def _make_motor_inhibition_cells(self, population_position, source, netcon_weight):
        for k, motor in enumerate(source):
            cell = Cell('minh', compile_paths="agents/commons/mods/sigma3syn")
            cell.name = "minh_%s[%s][%s]" % (population_position, cell.name, k)
            self.motor_inhibit.append(cell)
            soma = cell.add_sec("soma", diam=3, l=3, nseg=1)
            cell.insert("pas")
            cell.insert("hh")
            cell.add_synapse(source=motor.filter_secs("soma")(0.5), netcon_weight=netcon_weight,
                             seg=soma(0.5), mod_name="ExcSigma3Exp2Syn")
            for m, src in enumerate(source):
                if m == k: continue
                src.add_synapse(source=cell.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                                mod_name="Exp2Syn", e=-90)


def get_observation(x_train_obs):
    return cv2.normalize(cv2.resize(x_train_obs, (14, 14)), None, 0, 1.0, cv2.NORM_MINMAX)


def get_predicted_reward(output, true):
    predicted = -1
    reward = -1
    # todo if len() == 0, then probably there is no need to check the value?
    if len(output) == 1 and output[0].value > -1:
        predicted = output[0].index
        if predicted == true:
            reward = +1
    return predicted, reward


def update_true_predicted_correct_arr(last_true, last_predicted, correct_arr, predict_arr, true, predicted):
    last_true.append(true)
    last_predicted.append(predicted)
    if predicted >= 0:
        predict_arr[predicted] += 1
    if predicted == true:
        correct_arr[predicted] += 1


def rl_naive(agent, output, x_train, y_train, index, qval, stepsize, tries=100, alpha=0.5, gamma=0.9):
    # todo dwukrotne obliczanie tego samego get_predicted_reward(): tu i na koncu
    _, last_reward = get_predicted_reward(output=output, true=y_train[index])
    agent.reward_step(reward=last_reward, stepsize=stepsize)
    new_index = (index + 1) % y_train.shape[0]
    new_obs = get_observation(x_train_obs=x_train[new_index])
    new_output = agent.step(observation=new_obs, output_type="rate", epsilon=1)
    new_true = y_train[new_index]
    new_predicted, new_reward = get_predicted_reward(output=new_output, true=new_true)
    new_qval = (1 - alpha) * qval + alpha * (last_reward + gamma * new_reward)
    # todo nie musi zwracac agenta
    # todo zrobic z qval, by byl obliczany w queuenp; czy konieczne?
    return agent, new_output, new_true, new_predicted, new_reward, new_qval


def rl_sarsa(agent, output, x_train, y_train, index, qval, stepsize, tries=100, alpha=0.5, gamma=0.999):
    # todo dwukrotne obliczanie tego samego get_predicted_reward(): tu i na koncu
    _, last_reward = get_predicted_reward(output=output, true=y_train[index])
    # info run tries loops to evaluate the Q value for each of the possible actions
    actions = [-1, +1]
    q = np.zeros(len(actions))
    agent_list = [agent, copy.deepcopy(agent)]
    results = []
    new_output = new_true = new_predicted = new_reward = None
    for k, r in enumerate(actions):
        agent_list[k].reward_step(reward=r, stepsize=stepsize)
        this_gamma = 1
        for m in range(tries):
            p = (index + m + 1) % y_train.shape[0]
            new_obs = get_observation(x_train_obs=x_train[p])
            new_true = y_train[p]
            new_output = agent_list[-1].step(observation=new_obs, output_type="rate", epsilon=1)
            new_predicted, new_reward = get_predicted_reward(output=new_output, true=new_true)
            if m == 0:
                results.append([new_output, new_true, new_predicted, new_reward])
            this_gamma *= gamma
            q[k] += new_reward * this_gamma
    q = q / tries
    # info select agent with higher evaluated q
    best = q.index(max(q))
    agent = agent_list[best]
    new_qval = (1 - alpha) * qval + alpha * (last_reward + gamma * q[best])

    # info agent HAS to be passed back, since reference might have changed
    return agent, new_output, new_true, new_predicted, new_reward, new_qval


def main(display_interval, rl="naive"):
    x_train, y_train = mnist_prepare(num=MNIST_LABELS)
    # warning  use cv2 downsampling function: in agent build pass shapes _after_ downsampling
    # x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]

    kernel_size = 5
    padding = 3
    stride = 3
    np.random.seed(19283765)
    # info build the agent architecture
    agent = EbnerAgent(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE,
                       ach_tau=2, da_tau=50)
    # agent = EbnerAgentInhb(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbnerOlfactoryAgent(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbOlA(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    agent.build(input_shape=(x_train.shape[1] // 2, x_train.shape[2] // 2),
                x_kernel=Kernel(size=kernel_size, padding=padding, stride=stride),
                y_kernel=Kernel(size=kernel_size, padding=padding, stride=stride))
    agent.init(init_v=RESTING_POTENTIAL, warmup=int(display_interval * AGENT_STEPSIZE / DT), dt=DT)
    print("Agent {:s}\n\tinput cells: {:d}\n\tpixels per input: {:d}\n\tlearning with {} RL".format(
        agent.__class__.__name__, agent.input_cell_num, agent.x_kernel.size * agent.y_kernel.size, rl))
    run_params = MarkerParams(agent_class=agent.__class__.__name__, agent_stepsize=AGENT_STEPSIZE, dt=DT,
                              input_cell_num=INPUT_CELL_NUM, output_cell_num=MNIST_LABELS, output_labels=[0, 1, 2])

    # graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
    # graph.plot()
    # plt.draw()

    # %%
    correct_arr = np.zeros(MNIST_LABELS, dtype=int)
    predict_arr = np.zeros(MNIST_LABELS, dtype=int)
    last_true = []
    last_predicted = []
    alpha = 0.5
    gamma = 0.9
    predictions_queue: QueueNP = QueueNP(class_no=MNIST_LABELS, alpha=alpha, gamma=gamma)
    mean_time_start = time.time()
    # info a introductory run for the first example to initialize variables
    index = 0
    true = y_train[index]
    output = agent.step(observation=get_observation(x_train_obs=x_train[index]), output_type="rate", epsilon=1)
    predicted, reward = get_predicted_reward(output=output, true=true)
    qval = alpha * reward
    update_true_predicted_correct_arr(last_true=last_true, last_predicted=last_predicted, correct_arr=correct_arr,
                                      predict_arr=predict_arr, true=true, predicted=predicted)
    predictions_queue.push(r=reward, true=true, pred=predicted, qval=qval)
    processed = 1
    for index in itertools.cycle(range(0, y_train.shape[0])):
        if rl == "naive":
            agent, output, true, predicted, reward, qval = rl_naive(agent=agent, output=output, x_train=x_train,
                                                                    y_train=y_train, index=index, qval=qval,
                                                                    stepsize=AGENT_STEPSIZE)
        elif rl == "sarsa":
            agent, output, true, predicted, reward, qval = rl_sarsa(agent=agent, output=output, x_train=x_train,
                                                                    y_train=y_train, index=index, qval=qval,
                                                                    stepsize=AGENT_STEPSIZE, tries=10)
        else:
            raise ValueError("No other rl methods than naive or sarsa defined")
        processed += 1

        update_true_predicted_correct_arr(last_true=last_true, last_predicted=last_predicted, correct_arr=correct_arr,
                                          predict_arr=predict_arr, true=true, predicted=predicted)
        predictions_queue.push(r=reward, true=true, pred=predicted, qval=qval)
        if predicted == true:
            print_single_correct(processed=processed, true=true, correct_arr=correct_arr, predict_arr=predict_arr,
                                 predictions_queue=predictions_queue)

        if processed > 0 and processed % display_interval == 0:
            last_predicted = last_predicted[-display_interval:]
            last_true = last_true[-display_interval:]
            agent.rec_output.plot(animate=True,
                                  steps=int(AGENT_STEPSIZE / DT + 2 * display_interval * AGENT_STEPSIZE / DT),
                                  true_class=last_true, pred_class=last_predicted, show_true_predicted=True,
                                  marker_params=run_params)
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)
            time_elapsed = time.time() - mean_time_start
            print_summary(processed=processed, correct_arr=correct_arr, predict_arr=predict_arr,
                          predictions_queue=predictions_queue, time_elapsed=time_elapsed,
                          display_interval=display_interval)
            mean_time_start = time.time()


def print_single_correct(processed, true, correct_arr, predict_arr, predictions_queue):
    print("{:05d}: ---> {:2d}\t".format(processed, true), correct_arr, "/", predict_arr,
          "  [{}]".format(predictions_queue.mcc), "  [{}]".format(predictions_queue.f1),
          "  ({:.2f}%)".format(np.sum(correct_arr) / (processed + 1)),
          "  <{:+.2f}>".format(predictions_queue.qval)
          )


def print_summary(processed, correct_arr, predict_arr, predictions_queue, time_elapsed, display_interval):
    ratio = 3 * predict_arr / processed
    print("{:05d}  --->   \t".format(processed), correct_arr, "/", predict_arr,
          "/", correct_arr / predict_arr, "/", ratio,
          "  [{}]".format(predictions_queue.mcc), "  [{}]".format(predictions_queue.f1),
          "  ({:.2f})".format(np.sum(correct_arr) / processed),
          "  <{:.2f}>".format(predictions_queue.qval),
          "  {{{:.2f}s/it}}".format((time_elapsed) / display_interval)
          )


def mnist_prepare(num=10):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    index_list = []
    for i, labels in enumerate(y_train):
        if labels in list(np.arange(num)):
            index_list.append(i)

    x_train, y_train = x_train[index_list], y_train[index_list]
    return x_train, y_train


def make_imshow(x_train, x_pixel_size, y_pixel_size):
    fig, ax = plt.subplots(1, 1)

    x_ticks = np.arange(0, x_train.shape[1], x_pixel_size)
    y_ticks = np.arange(0, x_train.shape[2], y_pixel_size)
    ax.set_xticks(x_ticks)
    ax.set_xticks([i for i in range(x_train.shape[1])], minor=True)
    ax.set_yticks(y_ticks)
    ax.set_yticks([i for i in range(x_train.shape[2])], minor=True)

    obj = ax.imshow(x_train[0], cmap=plt.get_cmap('gray'), extent=[0, x_train.shape[1], 0, x_train.shape[2]])
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    return obj, ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", help="display interval", default=10, type=int)
    parser.add_argument("--rl", help="reinforcement method", choices=["naive", "sarsa"], default="naive", type=str)
    args = parser.parse_args()
    main(display_interval=args.display, rl=args.rl)
