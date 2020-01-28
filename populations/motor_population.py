from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population


class MotorPopuation(Population):

    def make_cell(self, **kwargs) -> Cell:
        cell = Cell("output")
        cell.make_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        cell.make_spike_detector()
        return cell

    def make_conn(self, cell: Cell, source, source_loc=None, weight=1, **kwargs) -> list:
        syns = cell.make_sypanses(source=source, weight=weight, mod_name="ExpSyn", target_sec="soma",
                                  source_loc=source_loc, target_loc=0.5, threshold=10, e=40, tau=3, delay=3)
        return syns
