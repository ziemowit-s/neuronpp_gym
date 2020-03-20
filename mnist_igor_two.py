import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuronpp.cells.cell import Cell
from neuronpp.utils.network_status_graph import NetworkStatusGraph

from agents.ebner_olfactory_agent import EbnerOlfactoryAgent
from agents.ebner_olfactory_agent import WEIGHT
from populations.ebner_hebbian_population import EbnerHebbianPopulation

# print(sys.path)
# sys.path.extend('/Users/igor/git/neuronpp/neuronpp')
for e in sys.path: print(e)
# print(sys.path)
found = False
for e in sys.path:
    if 'NEURON-7.7/nrn/lib/python' in e:
        found = True
        break
if not found:
    print("NEURON-7.7 path not found in sys.path; exiting")
    sys.exit(0)


class EbOlAg(EbnerOlfactoryAgent):
    def __init__(self, input_cell_num, input_size, output_size, input_max_hz, default_stepsize=20):
        self.hidden_cells = []
        self.inhibitory_cells = []
        super().__init__(input_cell_num=input_cell_num, input_size=input_size, output_size=output_size,
                         input_max_hz=input_max_hz, default_stepsize=default_stepsize)

    def _build_network(self, input_cell_num, input_size, output_cell_num):
        # INPUTS
        # info how many synapses per cell
        input_syn_per_cell = int(np.ceil(input_size / input_cell_num))
        # info generate new population of a given kind
        input_pop = EbnerHebbianPopulation("inp_0")
        # info create given number of cells in that population
        input_pop.create(cell_num=input_cell_num)
        # info connect each cell to given number of synapses with source equal to None (to be set to input image)
        input_pop.connect(source=None, syn_num_per_source=input_syn_per_cell, delay=1, netcon_weight=WEIGHT, rule='one')
        # info add mechanisms (?)
        input_pop.add_mechs(single_cell_mechs=self._add_mechs)

        # HIDDEN
        # info create the hidden population
        # todo here is a set number of cells; modify to use a parameter
        self.hidden_pop = self._make_modulatory_population(name="hid_1", cell_num=12, source=input_pop)
        # info assign cells to local variable
        self.hidden_cells = self.hidden_pop.cells

        # INHIBITORY NFB
        # info generate inhibitory cells (see local fun below)
        # todo again a given number of cells
        for i in range(4):
            # info probably each inhibitory is per three (3) cells in the hidden layer
            # todo only 6 inhibitory are inhibited, and these are overlapping <---- corrected
            # todo ??? the _sources_ of these cells are connected to the hidden neurons in population hid_1 as sources
            # todo shouldn't it be the OTHER_WAY_ROUND: the hid_1 population neurons should be the outputs for
            # todo the Inh_2 population inhibit them and act similarly to a bias in ANN's?
            self._make_inhibitory_cells(population_num=2, counter=i, sources=self.hidden_pop.cells[3 * i:3 * i + 3],
                                        netcon_weight=0.1)

        # OUTPUTS
        # info generate output population as a modulatory (?) one
        output_pop = self._make_modulatory_population("out_3", cell_num=output_cell_num, source=self.hidden_pop)

        return input_pop.cells, output_pop.cells

    def _make_inhibitory_cells(self, population_num, counter, netcon_weight, sources):
        """
        Generate a population of inhibitory cells and connect them
        :param population_num: number of the population, just for name (?)
        :param counter: identifier for the cell, just for name (?)
        :param netcon_weight: initial (?) weight (does it change?)
        :param sources: source to which it should be connected
        """
        # info generate a cell template
        cell = Cell('inh', compile_paths="agents/commons/mods/sigma3syn")
        cell.name = "Inh_%s[%s][%s]" % (population_num, cell.name, counter)
        # info append to list of inhibitory cells
        self.inhibitory_cells.append(cell)

        # info add soma: diam = ?, l = ?
        soma = cell.add_sec("soma", diam=5, l=5, nseg=1)
        # info add as passive cell (param mechanism_name in neuronpp/cells/section_cell.py:SectionCell.insert(mechanism_name, sec, *param)
        cell.insert('pas')
        cell.insert('hh')
        # info connect source; for each source
        for source in sources:
            # info add a synapse to cell
            cell.add_synapse(source=source.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                             mod_name="ExcSigma3Exp2Syn")
            # info add (connect) a synapse to source
            source.add_synapse(source=cell.filter_secs('soma')(0.5), netcon_weight=netcon_weight, seg=soma(0.5),
                               mod_name="Exp2Syn", e=-90)


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


AGENT_STEPSIZE = 50
MNIST_LABELS = 3
SKIP_PIXELS = 2
INPUT_CELL_NUM = 9
DT = 0.2


def main(display_interval):
    x_train, y_train = mnist_prepare(num=MNIST_LABELS)
    x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]
    input_size = x_train.shape[1] * x_train.shape[2]

    # info build the agent architecture
    # todo does recognising static characters using a RL learning make sense?
    # agent = EbnerOlfactoryAgent(input_cell_num=INPUT_CELL_NUM, input_size=input_size,
    #                             output_size=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    agent = EbOlAg(input_cell_num=INPUT_CELL_NUM, input_size=input_size,
                   output_size=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    agent.init(init_v=-80, warmup=10000, dt=DT)

    # warning what is the heatmap showing here???
    hitmap_shape = int(np.ceil(np.sqrt(INPUT_CELL_NUM)))
    # hitmap_graph = SpikesHeatmapGraph(name="MNIST heatmap", cells=agent.input_cells, shape=(hitmap_shape, hitmap_shape))

    # info get input size
    x_pixel_size, y_pixel_size = agent.get_input_cell_observation_shape(x_train[0])
    input_syn_per_cell = int(np.ceil(input_size / INPUT_CELL_NUM))
    print('pixels per input cell:', input_syn_per_cell)

    # todo display one beside the other, not over
    # obj, ax = make_imshow(x_train, x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size)

    agent_compute_time = 0
    agent_observe = True
    start_time = time.time()
    reset_time = time.time()
    gain = 0
    reward = None

    # the first to start from
    index = 0
    graph = NetworkStatusGraph(cells=[c for c in agent.cells if not "mot" in c.name])
    graph.plot()

    # %%
    correct_arr = np.zeros(MNIST_LABELS, dtype=int)
    predict_arr = np.zeros(MNIST_LABELS, dtype=int)
    processed = 0
    # info lists of last truue and predicted;
    last_true = []
    last_predicted = []
    while True:
        # info read the current input
        y = y_train[index]
        obs = x_train[index]

        # write time before agent step
        current_time_relative = (time.time() - agent_compute_time)
        # warning what do we need stepsize for?
        if agent_compute_time > 0:
            stepsize = current_time_relative * 1000
        else:
            stepsize = None

        # info compute the agent activation
        output = agent.step(observation=obs, output_type="rate", sort_func=lambda x: -x.value)[0]

        # info get the predicted value and compare with the correct one
        predicted = -1
        if output.value > -1:
            predicted = output.index
            predict_arr[predicted] += 1

        last_true.append(y)
        last_predicted.append(predicted)
        if predicted == y:
            reward = 1
            correct_arr[predicted] += 1
            print("{:05d}: recognized {:d}\t".format(processed, y), correct_arr, "/", predict_arr,
                  "\t({:.3f}%)".format(np.sum(correct_arr) / (processed + 1)))
        else:
            reward = -1
        # info reward the agent accordingly to the output
        # todo look carefully inside
        agent.make_reward_step(reward=reward)

        # write time after agent step
        agent_compute_time = time.time()

        last_predicted = last_predicted[-display_interval:]
        last_true = last_true[-display_interval:]
        if processed > 0 and processed % display_interval == 0:
            # info update input image image
            # obj.set_data(obs)
            # ax.set_title('Predicted: %s True: %s' % (predicted, y))
            # info update weights in graph
            # graph.update_spikes(agent.sim.t)
            # graph.update_weights('w')
            # info update heatmap
            # hitmap_graph.plot()
            plt.draw()
            plt.pause(1e-9)
            # agent.rec_input.plot(animate=True, position=(4, 4))
            # info display output act for last display_interval examples
            # info left AGENT_STEPSIZE / DT for additional steps on the left of display
            agent.rec_output.plot(animate=True,
                                  steps=int(AGENT_STEPSIZE / DT + 2 * display_interval * AGENT_STEPSIZE / DT),
                                  true_class=last_true, pred_class=last_predicted, stepsize=AGENT_STEPSIZE, dt=DT)

        index += 1
        processed += 1
        if index == y_train.shape[0]:
            index = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", help="display interval", default=10, type=int)
    args = parser.parse_args()
    main(display_interval=args.display)
