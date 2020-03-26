import argparse
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.cells.cell import Cell
from neuronpp.utils.network_status_graph import NetworkStatusGraph

from agents.agent import Kernel
from agents.ebner_agent import EbnerAgent
from agents.ebner_olfactory_agent import EbnerOlfactoryAgent

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

AGENT_STEPSIZE = 60
MNIST_LABELS = 3
SKIP_PIXELS = 2
INPUT_CELL_NUM = 9
DT = 0.2
RESTING_POTENTIAL = -80


class EbOlA(EbnerOlfactoryAgent):
    def _build_network(self, input_cell_num, input_size, output_cell_num):
        input_pop, output_pop = super()._build_network(input_cell_num=input_cell_num, input_size=input_size,
                                                       output_cell_num=output_cell_num)
        # info add some inhibitory networks to the output layer
        self.motor_inhibit = []
        self._make_motor_inhibition_cells(population_position="minh_4", source=output_pop, netcon_weight=0.1)
        return input_pop, output_pop

    def _make_motor_inhibition_cells(self, population_position, source, netcon_weight):
        for k, motor in enumerate(source):
            cell = Cell('minh', compile_paths="agents/commons/mods/sigma3syn")
            cell.name = "Minh_%s[%s][%s]" % (population_position, cell.name, k)
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


def main(display_interval):
    x_train, y_train = mnist_prepare(num=MNIST_LABELS)
    # warning  use cv2 downsampling function: in agent build pass shapes _after_ downsampling
    # x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]

    # info build the agent architecture
    agent = EbnerAgent(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbnerOlfactoryAgent(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbOlA(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    kernel_size = 5
    padding = 3
    stride = 3
    agent.build(input_shape=(x_train.shape[1] // 2, x_train.shape[2] // 2),
                x_kernel=Kernel(size=kernel_size, padding=padding, stride=stride),
                y_kernel=Kernel(size=kernel_size, padding=padding, stride=stride))
    agent.init(init_v=RESTING_POTENTIAL, warmup=int(2 * display_interval * AGENT_STEPSIZE / DT), dt=DT)
    print("Agent {:s}\n\tinput cell number: {:d}\n\tpixels per input cell: ???".format(agent.__class__.__name__,
                                                                                       agent.input_cell_num))  # , agent.input_syn_per_cell)

    # obj, ax = make_imshow(x_train, x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size)
    heatmap_shape = int(np.ceil(np.sqrt(INPUT_CELL_NUM)))
    # heatmap_graph = SpikesHeatmapGraph(name="MNIST heatmap", cells=agent.input_cells, shape=(heatmap_shape, heatmap_shape))

    agent_observe = True
    start_time = time.time()
    reset_time = time.time()
    gain = 0
    reward = None

    graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
    graph.plot()
    plt.draw()

    # %%
    agent_compute_time = 0
    # the first to start from
    index = 0
    correct_arr = np.zeros(MNIST_LABELS, dtype=int)
    predict_arr = np.zeros(MNIST_LABELS, dtype=int)
    processed = 0
    last_true = []
    last_predicted = []
    while True:
        y = y_train[index]
        # downsample input
        # obs = x_train[index]
        obs = cv2.normalize(cv2.resize(x_train[index], (14, 14)), None, 0, 1.0, cv2.NORM_MINMAX)

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)
        # warning what do we need stepsize for?
        if agent_compute_time > 0:
            stepsize = current_time_relative * 1000
        else:
            stepsize = None

        output = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value)
        predicted = -1
        if len(output) == 1 and output[0].value > -1:
            predicted = output[0].index
            predict_arr[predicted] += 1

        last_true.append(y)
        last_predicted.append(predicted)
        if predicted == y:
            reward = 1
            correct_arr[predicted] += 1
            print("{:05d}: recognized {:5d}\t".format(processed, y), correct_arr, "/", predict_arr,
                  "\t({:.3f}%)".format(np.sum(correct_arr) / (processed + 1)))
        else:
            reward = -1
        agent.reward_step(reward=reward, stepsize=AGENT_STEPSIZE)

        # write time after agent step
        agent_compute_time = time.time()

        last_predicted = last_predicted[-display_interval:]
        last_true = last_true[-display_interval:]
        if processed > 0 and processed % display_interval == 0:
            # obj.set_data(obs)
            # ax.set_title('Predicted: %s True: %s' % (predicted, y))
            # graph.update_spikes(agent.sim.t)
            # graph.update_weights('w')
            # hitmap_graph.plot()
            # agent.rec_input.plot(animate=True, position=(4, 4))
            # info display output act for last display_interval examples
            # info left AGENT_STEPSIZE / DT for additional steps on the left of display
            agent.rec_output.plot(animate=True,
                                  steps=int(AGENT_STEPSIZE / DT + 2 * display_interval * AGENT_STEPSIZE / DT),
                                  true_class=last_true, pred_class=last_predicted, stepsize=AGENT_STEPSIZE, dt=DT,
                                  show_true_predicted=True, true_labels=[0, 1, 2])
            plt.draw()
            plt.pause(1)
            print("{:05d}               \t".format(processed), correct_arr, "/", predict_arr,
                  "\t({:.3f}%)".format(np.sum(correct_arr) / (processed + 1)))

        index += 1
        processed += 1
        if index == y_train.shape[0]:
            index = 0


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
    args = parser.parse_args()
    main(display_interval=args.display)
