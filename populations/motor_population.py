from neuronpp.cells.cell import Cell
from neuronpp.core.populations.population import Population


class MotorPopulation(Population):

    def cell_definition(self, **kwargs) -> Cell:
        cell = Cell("output")
        cell.add_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        cell.make_spike_detector(cell.filter_secs("soma")(0.5))
        return cell

    def syn_definition(self, cell: Cell, source, netcon_weight=1, **kwargs) -> list:
        soma = cell.filter_secs("soma")
        syns = cell.add_sypanse(source=source, netcon_weight=netcon_weight, mod_name="ExpSyn",
                                seg=soma(0.5), threshold=10, e=40, tau=3, delay=3)
        return syns
