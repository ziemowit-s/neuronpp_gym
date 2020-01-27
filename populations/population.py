import abc

from neuronpp.cells.cell import Cell
from neuronpp.utils.record import Record


class Population:
    def __init__(self):
        self.cell_counter = 0
        self.cells = []
        self.syns = []
        self.recs = {}

    def create(self, cell_num, **kwargs):
        result = []
        for i in range(cell_num):
            cell = self._create_single_cell(**kwargs)
            self.cell_counter += 1
            self.cells.append(cell)
            result.append(cell)

        return result

    def add_mechs(self, single_cell_mechs):
        for cell in self.cells:
            single_cell_mechs(cell)

    def connect(self, sources, sec_name="soma", loc=0.5, rule="all", **kwargs):
        """

        :param sources:
            int for empty synapses, or Cells for real connections
        :param sec_name:
        :param loc:
        :param rule:
            'all' - all-to-all connections
            'one' - one-to-one connections
        :return:
            list of list of synapses
        """
        result = []
        if sources is None:
            sources = [None for _ in range(len(self.cells))]

        elif isinstance(sources, int):
            sources = [None for _ in range(sources)]

        if rule == 'all':
            for source_cell in sources:
                for cell in self.cells:
                    syns = self._conn(source_cell, sec_name, cell, loc, **kwargs)
                    result.append(syns)

        elif rule == 'one':
            for source_cell, cell in zip(sources, self.cells):
                syns = self._conn(source_cell, sec_name, cell, loc, **kwargs)
                result.append(syns)
        else:
            raise TypeError("The only allowed rules are 'all' or 'one', but provided rule '%s'" % rule)

        self.syns.extend(result)
        return result

    def record(self, sec_name="soma", loc=0.5, variable='v'):
        d = [cell.filter_secs(sec_name)[0] for cell in self.cells]
        rec = Record(d, locs=loc, variables=variable)
        self.recs[variable] = rec

    def plot(self, steps=10000, y_lim=(-80, 50), position=None):
        """
        Plots each recorded variable for each neurons in the population.

        :param steps:
            how many timesteps to see on the graph
        :param y_lim:
            tuple of limits for y axis. Default is (-80, 50)
        :param position:
            position of all subplots ON EACH figure (each figure is created for each variable separately).
            * position=(3,3) -> if you have 9 neurons and want to display 'v' on 3x3 matrix
            * position='merge' -> it will display all figures on the same graph.
            * position=None -> Default, each neuron has separated  axis (row) on the figure.
        """
        for r in self.recs.values():
            r.plot(steps=steps, y_lim=y_lim, position=position)

    def _conn(self, source_cell, sec_name, cell, loc, **kwargs):
        if source_cell is None:
            source_section = None
        else:
            source_section = source_cell.filter_secs(sec_name)[0]
        syns = self._connect_sigle_cell(source_section=source_section, cell=cell, loc=loc, **kwargs)

        return syns

    @abc.abstractmethod
    def _connect_sigle_cell(self, source_section, cell, loc, **kwargs) -> list:
        """
        Must return syns list.
        :param cell:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_single_cell(self, **kwargs) -> Cell:
        """
        Must return single cell.
        :return:
        """
        raise NotImplementedError()