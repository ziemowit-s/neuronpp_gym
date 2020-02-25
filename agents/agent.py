import abc
from neuronpp.core.cells.core_cell import CoreCell
from neuronpp.core.populations.population import Population
from neuronpp.utils.utils import show_connectivity_graph


class Agent:

    def show_connectivity_graph(self):
        cells = []
        for v in self.__dict__.values():
            cs = self._make_connectivity(v)
            cells.extend(cs)
        show_connectivity_graph(cells)

    @abc.abstractmethod
    def step(self, observation=None, reward=None):
        raise NotImplementedError()

    def _make_connectivity(self, v):
        acc = []
        if isinstance(v, CoreCell):
            acc.append(v)
        elif isinstance(v, Population):
            acc.extend(v.cells)
        elif isinstance(v, list):
            for vv in v:
                ac = self._make_connectivity(vv)
                acc.extend(ac)
        return acc
